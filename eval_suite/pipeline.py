"""
eval_suite/pipeline.py
======================
Esecuzione del sistema durante la valutazione.

Perché serve un modulo dedicato invece di invocare il grafo direttamente:
gli scorer `RetrievalRelevance` e `RetrievalGroundedness` di MLflow **non
leggono gli argomenti della funzione valutata, leggono la traccia**. Per
trovare il contesto recuperato cercano nella traccia uno span di tipo
`RETRIEVER` e ne interpretano l'output come elenco di documenti.

Il grafo dell'applicazione non produce span di quel tipo, e modificarlo per
farlo significherebbe cambiare il codice di produzione per un'esigenza di
misura. Qui viene invece registrato uno span dedicato all'interno della sola
traccia di valutazione: il sistema resta intatto e la strumentazione appartiene
a chi misura.
"""
import logging
import threading
import time
import uuid
from typing import Any, Dict, List

import mlflow
from mlflow.entities import Document, SpanType

logger = logging.getLogger(__name__)

# Contatore di avanzamento, condiviso fra i thread che MLflow usa per eseguire
# i casi in parallelo. Serve un lock perché l'incremento e la lettura devono
# essere atomici: senza, due casi conclusi nello stesso istante stamperebbero
# lo stesso numero.
_avanzamento = {"fatti": 0, "totale": 0, "lock": threading.Lock()}


def inizia(totale: int) -> None:
    """Azzera il contatore prima di una suite, fissandone il numero di casi."""
    with _avanzamento["lock"]:
        _avanzamento["fatti"] = 0
        _avanzamento["totale"] = totale


def _segnala_avanzamento(query: str, escalato: bool, durata: float) -> None:
    """
    Emette una riga di avanzamento a ogni caso concluso.

    La barra che MLflow disegna da sé viene riscritta sulla stessa riga con
    caratteri di controllo: nei log di Docker, che non sono un terminale,
    risulta illeggibile o del tutto assente. Qui si emette invece una riga per
    caso, con il tempo impiegato.

    Serve soprattutto a distinguere due situazioni che dall'esterno appaiono
    identiche: un sistema lento — perché sta attendendo fra un tentativo e
    l'altro dopo un errore di quota — e un sistema bloccato. Se le righe
    continuano ad apparire, per quanto distanziate, sta procedendo.
    """
    with _avanzamento["lock"]:
        _avanzamento["fatti"] += 1
        fatti, totale = _avanzamento["fatti"], _avanzamento["totale"]

    quota = fatti / totale if totale else 0.0
    riempiti = int(quota * 24)
    barra = "█" * riempiti + "·" * (24 - riempiti)
    logger.info("[%s] %3d/%-3d %5.1f%% · %5.1fs · %s · %s",
                barra, fatti, totale, quota * 100, durata,
                "escalato   " if escalato else "risolto     ",
                query.replace("\n", " ")[:48])


@mlflow.trace(name="retrieved_context", span_type=SpanType.RETRIEVER)
def _registra_contesto(documenti: List[Document]) -> List[Document]:
    """
    Registra il contesto recuperato come span `RETRIEVER` nella traccia.

    Non esegue alcun recupero: il recupero è già avvenuto dentro il grafo. Lo
    scopo è unicamente rendere quel contesto leggibile agli scorer, che
    altrimenti riporterebbero «nessun contesto trovato nella traccia».

    Ogni documento porta `doc_uri` fra i metadati, cioè l'identificativo della
    policy o del ticket di provenienza: è ciò che permette, leggendo un
    punteggio basso nell'interfaccia, di risalire al documento che l'ha
    causato invece di vedere solo un numero.
    """
    return documenti


def _documenti(esito: Dict[str, Any]) -> List[Document]:
    """
    Converte il contesto recuperato nel formato atteso da MLflow.

    Il testo dei ticket viene composto con `ticket_as_context`, la stessa
    funzione usata per costruire il prompt: gli scorer devono giudicare
    esattamente ciò che il modello ha letto, non una riformulazione. Se le due
    rappresentazioni divergessero, un punteggio di fondatezza basso non
    direbbe se a sbagliare è il sistema o la misura.
    """
    from app.retrieval import ticket_as_context

    docs = [
        Document(page_content=d.get("text", ""),
                 metadata={"doc_uri": d.get("source", "?"), "kind": "policy"})
        for d in esito.get("kb_docs", [])
    ]
    docs += [
        Document(page_content=ticket_as_context(t),
                 metadata={"doc_uri": t.get("source", "?"), "kind": "past_ticket"})
        for t in esito.get("kb_tickets", [])
    ]
    return docs


@mlflow.trace(name="helpdesk_agent")
def predici(query: str, ticket_id: str = "") -> Dict[str, Any]:
    """
    Esegue il sistema su una richiesta e ne restituisce l'esito.

    La funzione è tracciata esplicitamente perché deve costituire la **radice**
    della traccia: senza, lo span di registrazione del contesto resterebbe
    l'unico span di primo livello e MLflow assumerebbe come esito della
    valutazione l'elenco dei documenti anziché la risposta del sistema.

    MLflow invoca questa funzione per ogni riga del dataset, passandole i campi
    di `inputs`, e la traccia risultante è ciò su cui operano gli scorer.

    Viene invocato il **grafo completo**, non le singole funzioni: si misura il
    sistema come si comporta in esercizio, cablaggio compreso. Se la decisione
    è di escalare, il grafo si sospende su `interrupt()` e l'esito è nel
    payload; altrimenti arriva a `finalize` e l'esito è nello stato.

    `ticket_id` attiva il leave-one-out ed è vuoto per i casi scritti a mano,
    che non sono ticket indicizzati. L'esclusione non serve solo a evitare che
    una richiesta recuperi sé stessa: il payload di un ticket contiene l'esito
    dell'escalation, quindi senza esclusione il modello leggerebbe nel contesto
    la risposta alla domanda che gli si sta ponendo.

    Ogni chiamata usa un identificativo di conversazione nuovo: i casi sono
    indipendenti e non devono ereditare la cronologia l'uno dell'altro.
    """
    from app.graph import graph

    avvio = time.time()
    stato = {"user_query": query, "exclude_sources": [ticket_id] if ticket_id else []}
    out = graph.invoke(stato, {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}})

    if "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        esito = {
            "escalated": True,
            "answer": payload.get("draft_answer", ""),
            "triggers": [t["code"] for t in payload.get("triggers", [])],
            "kb_docs": payload.get("kb_docs", []),
            "kb_tickets": payload.get("kb_tickets", []),
        }
    else:
        esito = {
            "escalated": False,
            "answer": out.get("final_answer", ""),
            "triggers": [],
            "kb_docs": out.get("kb_docs_context", []),
            "kb_tickets": out.get("kb_tickets_context", []),
        }

    _registra_contesto(_documenti(esito))
    _segnala_avanzamento(query, esito["escalated"], time.time() - avvio)

    # Gli scorer che valutano la risposta ricevono `outputs`: si restituisce la
    # sola risposta come stringa sotto una chiave esplicita, insieme ai campi
    # che servono alla valutazione dell'escalation.
    return {
        "response": esito["answer"],
        "escalated": esito["escalated"],
        "triggers": esito["triggers"],
    }
