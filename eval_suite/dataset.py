"""
eval_suite/dataset.py
=====================
Costruzione dei casi di valutazione ed esecuzione del sistema su di essi.

Separare questo modulo dagli scorer risponde a una distinzione precisa: qui si
decide **su cosa** misurare, negli scorer **come**. Le due cose cambiano per
ragioni diverse — il dataset cresce quando si aggiungono casi, gli scorer
cambiano quando si rivede un criterio — e tenerle insieme obbligherebbe a
rimettere mano allo stesso file per motivi scollegati.
"""
import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

KB = Path(__file__).parent.parent / "app" / "knowledge_base"


def carica_casi(sample: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Costruisce i casi di valutazione dallo storico dei ticket.

    Ogni riga ha la forma attesa da `mlflow.genai.evaluate`: `inputs` è ciò che
    il sistema riceve, `expectations` la verità di riferimento contro cui si
    misura. I metadati (`case_id`, `subcategory`) restano fuori da entrambi
    perché servono a leggere i risultati, non a produrli.

    `expectations` contiene la sola decisione attesa di escalation: è l'unica
    verità di riferimento di cui disponiamo senza annotazione manuale, dato che
    `was_escalated_to_human` registra ciò che un operatore umano decise davvero
    all'epoca.

    Il campionamento è stratificato per sottocategoria e con seme fisso. La
    stratificazione evita che le sottocategorie meno numerose — cinque ticket
    su centotrentacinque — spariscano dal campione, che sono proprio quelle su
    cui il sistema ha più probabilità di sbagliare. Il seme fisso serve a
    rendere confrontabili due esecuzioni: se il campione cambiasse a ogni
    lancio, le differenze fra due run sarebbero rumore indistinguibile da un
    effetto reale.
    """
    tickets = json.loads((KB / "past_tickets.json").read_text(encoding="utf-8"))
    casi = [
        {
            "case_id": t["ticket_id"],
            "subcategory": t["subcategory"],
            "inputs": {
                "query": f"{t['subject']}\n\n{t['description']}",
                # Necessario al leave-one-out: vedi `esegui_sistema`.
                "ticket_id": t["ticket_id"],
            },
            "expectations": {"escalate": bool(t["was_escalated_to_human"])},
        }
        for t in tickets
    ]
    return _campione_stratificato(casi, sample, seed) if sample else casi


def _campione_stratificato(casi: List[Dict[str, Any]], n: int,
                           seed: int) -> List[Dict[str, Any]]:
    """
    Sottocampione che preserva le proporzioni per sottocategoria.

    Ogni gruppo contribuisce con almeno un caso, quindi il campione non può
    scendere sotto il numero di sottocategorie: chiedendone meno se ne
    ottengono comunque venti. È voluto, ma va saputo perché ogni caso costa
    più chiamate al modello giudice.
    """
    if n <= 0 or n >= len(casi):
        return casi

    gruppi: Dict[str, List[Dict[str, Any]]] = {}
    for c in casi:
        gruppi.setdefault(c["subcategory"], []).append(c)

    rng = random.Random(seed)
    campione: List[Dict[str, Any]] = []
    for _, gruppo in sorted(gruppi.items()):
        quota = max(1, round(n * len(gruppo) / len(casi)))
        campione.extend(rng.sample(gruppo, min(quota, len(gruppo))))
    rng.shuffle(campione)
    return campione


def esegui_sistema(query: str, ticket_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Esegue il sistema reale su una richiesta e restituisce quanto serve agli
    scorer.

    È la `predict_fn` che MLflow invoca per ogni riga del dataset. Si invoca il
    **grafo completo** e non le singole funzioni, così la misura riguarda il
    sistema com'è in esercizio, cablaggio compreso: un errore
    nell'orchestrazione verrebbe altrimenti nascosto proprio dalla valutazione
    che dovrebbe scoprirlo.

    Quando la decisione è di escalare, il grafo si sospende su `interrupt()` e
    `invoke` ritorna con la chiave `__interrupt__`; altrimenti l'esito è nello
    stato. Le due strade vengono normalizzate nella stessa forma, perché agli
    scorer non interessa quale ramo sia stato percorso.

    Il **contesto è restituito come elenco di testi separati** e non
    concatenato: la pertinenza si valuta chunk per chunk, e un unico blocco
    renderebbe la misura cieca a quale porzione sia utile — che è proprio
    l'informazione necessaria per capire se il ranking funziona.

    Il **leave-one-out** esclude dal recupero il ticket usato come query.
    Senza, ogni ticket recupererebbe sé stesso con similarità prossima a uno, e
    il modello leggerebbe nel contesto l'esito che gli stiamo chiedendo di
    prevedere: misureremmo la capacità di copiare, non di decidere.

    Ogni chiamata usa un `thread_id` nuovo perché i casi devono essere
    indipendenti e non ereditare la cronologia l'uno dell'altro.
    """
    from app.graph import graph
    from app.retrieval import ticket_as_context

    stato = {"user_query": query, "exclude_sources": [ticket_id] if ticket_id else []}
    out = graph.invoke(stato, {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}})

    interrotto = "__interrupt__" in out
    payload = out["__interrupt__"][0].value if interrotto else {}

    docs = out.get("kb_docs_context", []) or []
    tickets = out.get("kb_tickets_context", []) or []

    return {
        "escalated": interrotto,
        "answer": payload.get("draft_answer") if interrotto else out.get("final_answer", ""),
        # I due tipi di fonte sono uniformati a semplici stringhe: le policy
        # hanno già un testo pronto, i ticket sono conservati integri e vengono
        # resi leggibili dallo stesso formattatore usato per il prompt, così il
        # testo giudicato è quello che il modello ha effettivamente letto.
        "context": [d.get("text", "") for d in docs]
                   + [ticket_as_context(t) for t in tickets],
    }
