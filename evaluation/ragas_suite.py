"""
evaluation/ragas_suite.py
=========================
Valutazione della pipeline RAG con le metriche **RAGAS**, eseguite attraverso
l'harness di valutazione di **MLflow**.

    python -m evaluation.ragas_suite --sample 20

Perché passare da `mlflow.genai.evaluate` invece di chiamare `ragas.evaluate`
-----------------------------------------------------------------------------
Le due librerie fanno cose diverse e non alternative. RAGAS fornisce le
**procedure di misura**; MLflow fornisce l'**infrastruttura di valutazione**:
esecuzione sul dataset, raccolta dei risultati per riga, collegamento alle
tracce, e persistenza in un formato confrontabile fra run.

Chiamare direttamente `ragas.evaluate` produrrebbe un DataFrame che poi
andrebbe registrato a mano, come artefatto separato e slegato dalle tracce.
Avvolgendo invece ogni metrica RAGAS in uno **scorer MLflow**, si ottiene:

- ogni punteggio diventa un *assessment* attaccato alla traccia della singola
  esecuzione, quindi in interfaccia si passa dal numero al caso che l'ha
  prodotto senza cercare corrispondenze a mano;
- gli scorer RAGAS e quelli nativi di MLflow (`RetrievalGroundedness`,
  `RelevanceToQuery`) girano nella **stessa** valutazione, sugli stessi dati,
  e sono confrontabili direttamente;
- il collegamento con la versione dei prompt e la configurazione avviene
  automaticamente, essendo parte del run.

La motivazione di fondo: RAGAS misura, MLflow rende la misura tracciabile.
Sostituire l'uno con l'altro significherebbe rinunciare a metà del valore.

Il ruolo di `Feedback`
----------------------
Ogni scorer restituisce un oggetto `Feedback` e non un semplice numero. Serve
a portare con sé la **motivazione** e la fonte del giudizio: con metriche
calcolate da un LLM, un punteggio anomalo può dipendere tanto dal sistema
valutato quanto da un giudizio sbagliato, e senza la motivazione i due casi
sono indistinguibili.

Dipendenze
----------
RAGAS sta in un extra opzionale perché richiede `langchain-community<0.4`:
vincolare l'ambiente del servizio a quella versione per una libreria usata
solo in valutazione sarebbe un accoppiamento ingiustificato.

    uv sync --extra ragas
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app import config
from app.retrieval import ticket_as_context
from evaluation.run_evaluation import run_system, stratified_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ragas-suite")

KB = Path(__file__).parent.parent / "app" / "knowledge_base"

# Istanze delle metriche RAGAS, costruite una sola volta e riusate da tutti
# gli scorer: ciascuna incapsula il proprio modello giudice, e ricrearle a
# ogni riga significherebbe ricostruire il client a ogni caso valutato.
_metriche: Dict[str, Any] = {}


# ==========================================================================
# Costruzione dei modelli e delle metriche
# ==========================================================================

def costruisci_metriche() -> Dict[str, Any]:
    """
    Istanzia le quattro metriche RAGAS con il provider Gemini come giudice.

    Nelle metriche della famiglia `collections` il modello si passa al
    **costruttore**, non alla funzione di valutazione: ogni metrica incapsula
    il proprio giudice. La conseguenza pratica è che si potrebbero usare
    modelli diversi per criteri diversi — per esempio uno più capace per la
    fondatezza, che è il criterio più delicato.

    Si riusa il client `google-genai` già impiegato dall'applicazione, senza
    passare da adattatori LangChain: RAGAS lo accetta direttamente.

    Nota metodologica: `JUDGE_MODEL` è configurabile separatamente proprio
    perché usare come giudice lo stesso modello che genera le risposte riduce
    l'indipendenza del giudizio — valutato e valutatore condividono gli stessi
    punti ciechi.
    """
    global _metriche
    if _metriche:
        return _metriche

    from google import genai
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
    )

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    llm = llm_factory(config.JUDGE_MODEL, provider="google", client=client)
    emb = GoogleEmbeddings(client=client, model=config.GEMINI_EMBEDDING_MODEL)

    _metriche = {
        # --- qualità del RECUPERO ---
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        # --- qualità della GENERAZIONE ---
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
    }
    return _metriche


def _valuta(nome: str, **kwargs) -> Optional[float]:
    """
    Esegue una metrica RAGAS restituendo il punteggio, oppure None se fallisce.

    Le metriche RAGAS espongono un'interfaccia asincrona: qui viene eseguita
    in modo sincrono perché gli scorer di MLflow sono chiamati in un contesto
    sincrono, e mescolare i due modelli complicherebbe il codice senza
    beneficio a queste dimensioni di campione.

    Un fallimento su una singola riga ritorna `None` invece di propagare
    l'eccezione: interrompere l'intera valutazione perché un caso su venti ha
    prodotto una risposta che il giudice non riesce a interpretare sarebbe
    sproporzionato. Il caso viene semplicemente escluso dalla media.
    """
    try:
        metrica = costruisci_metriche()[nome]
        risultato = asyncio.run(metrica.ascore(**kwargs))
        return float(risultato.value)
    except Exception:
        logger.warning("Metrica '%s' non calcolabile su questo caso.", nome, exc_info=True)
        return None


def _feedback(nome: str, valore: Optional[float], motivazione: str):
    """
    Confeziona un punteggio come `Feedback` MLflow.

    La fonte è dichiarata come giudizio di un modello, non come misura
    deterministica: in interfaccia distingue questi punteggi da quelli
    calcolati con formule chiuse, distinzione che conta quando si interpreta
    un valore anomalo.
    """
    from mlflow.entities import AssessmentSource, Feedback

    return Feedback(
        name=nome,
        value=valore,
        rationale=motivazione,
        source=AssessmentSource(source_type="LLM_JUDGE",
                                source_id=f"ragas:/{config.JUDGE_MODEL}"),
    )


# ==========================================================================
# Gli scorer: una metrica RAGAS ciascuno, nell'interfaccia di MLflow
# ==========================================================================
# Ogni scorer riceve da MLflow i campi del dataset e restituisce un Feedback.
# La firma dichiara quali campi servono: MLflow passa solo quelli richiesti.

def costruisci_scorer() -> List[Any]:
    """
    Definisce i quattro scorer che avvolgono le metriche RAGAS.

    Sono creati dentro una funzione, e non a livello di modulo, perché il
    decoratore `@scorer` richiede MLflow importato: definendoli al modulo, il
    solo import di questo file fallirebbe in un ambiente privo dell'extra
    `ragas`, impedendo persino di leggere l'help da riga di comando.
    """
    from mlflow.genai.scorers import scorer

    @scorer
    def ragas_context_precision(inputs, outputs):
        """
        RECUPERO — i chunk recuperati sono pertinenti alla richiesta?

        Nella variante *without reference* la pertinenza è giudicata rispetto
        alla richiesta e alla risposta prodotta, senza bisogno di una verità
        di riferimento. È la variante adatta qui: annotare a mano quali chunk
        siano rilevanti per 135 ticket non è sostenibile.

        Tiene conto della **posizione** nel ranking: un chunk pertinente in
        prima posizione vale più dello stesso chunk in terza. È coerente con
        come il contesto viene poi usato, dato che al modello i primi chunk
        arrivano con maggiore rilievo.
        """
        valore = _valuta("context_precision",
                         user_input=inputs["query"],
                         retrieved_contexts=outputs["contexts"],
                         response=outputs["answer"])
        return _feedback("ragas_context_precision", valore,
                         "Pertinenza dei chunk recuperati, pesata per posizione nel ranking.")

    @scorer
    def ragas_context_recall(inputs, outputs, expectations):
        """
        RECUPERO — il contesto recuperato copre quanto serve per rispondere?

        È l'unica delle quattro che usa un riferimento, qui la sintesi di come
        il ticket fu effettivamente risolto. L'uso è legittimo e va distinto
        da quello scartato altrove nel progetto: **non** si confronta la
        risposta con il riferimento — sarebbero testi con destinatari diversi,
        uno rivolto all'utente e uno annotato a posteriori — ma si verifica se
        il contesto recuperato contiene le informazioni che il riferimento
        cita. È esattamente ciò che il recupero avrebbe dovuto trovare.
        """
        valore = _valuta("context_recall",
                         user_input=inputs["query"],
                         retrieved_contexts=outputs["contexts"],
                         reference=expectations["reference"])
        return _feedback("ragas_context_recall", valore,
                         "Copertura del riferimento da parte del contesto recuperato.")

    @scorer
    def ragas_faithfulness(inputs, outputs):
        """
        GENERAZIONE — ogni affermazione della risposta è sostenuta dal contesto?

        Scompone la risposta in affermazioni elementari e verifica ciascuna
        contro il contesto. È la metrica che intercetta le procedure aziendali
        inventate, il rischio specifico introdotto dall'istruzione «risolvi in
        un solo messaggio»: un modello spinto a non fare domande può colmare i
        vuoti con passaggi plausibili ma inesistenti.

        Va letta insieme a `context_recall`: fondatezza bassa con copertura
        alta indica che il modello inventa pur avendo il materiale; entrambe
        basse indicano che il problema è a monte, nel recupero.
        """
        valore = _valuta("faithfulness",
                         user_input=inputs["query"],
                         retrieved_contexts=outputs["contexts"],
                         response=outputs["answer"])
        return _feedback("ragas_faithfulness", valore,
                         "Quota di affermazioni della risposta sostenute dal contesto.")

    @scorer
    def ragas_answer_relevancy(inputs, outputs):
        """
        GENERAZIONE — la risposta affronta davvero la domanda posta?

        Procede al contrario: dalla risposta genera le domande a cui essa
        risponderebbe, e ne misura la somiglianza con quella originale. Per
        questo richiede anche un modello di embedding, oltre al giudice.

        Penalizza le risposte evasive o generiche, che è una verifica utile in
        un help desk: una risposta corretta ma che non affronta il problema
        specifico costringe comunque l'utente a riscrivere.
        """
        valore = _valuta("answer_relevancy",
                         user_input=inputs["query"],
                         response=outputs["answer"])
        return _feedback("ragas_answer_relevancy", valore,
                         "Aderenza della risposta alla domanda effettivamente posta.")

    return [ragas_context_precision, ragas_context_recall,
            ragas_faithfulness, ragas_answer_relevancy]


# ==========================================================================
# Dataset e funzione di predizione
# ==========================================================================

def carica_casi(sample: int) -> List[Dict[str, Any]]:
    """
    Costruisce il dataset di valutazione nel formato atteso da MLflow.

    Ogni riga ha tre chiavi: `inputs` (ciò che il sistema riceve),
    `expectations` (la verità di riferimento) e i metadati utili a leggere i
    risultati. MLflow passa poi a ciascuno scorer solo i campi che la sua
    firma dichiara.

    Il campione è stratificato per sottocategoria con seme fisso, come nelle
    altre suite: due esecuzioni devono valutare gli stessi casi, altrimenti le
    differenze fra run sarebbero rumore indistinguibile da un effetto reale.
    """
    tickets = json.loads((KB / "past_tickets.json").read_text(encoding="utf-8"))
    casi = [
        {
            "case_id": t["ticket_id"],
            "subcategory": t["subcategory"],
            "inputs": {
                "query": f"{t['subject']}\n\n{t['description']}",
                # Serve alla funzione di predizione per il leave-one-out:
                # un ticket usato come query non deve recuperare sé stesso.
                "ticket_id": t["ticket_id"],
            },
            "expectations": {"reference": t["resolution_summary"]},
        }
        for t in tickets
    ]
    return stratified_sample(casi, sample) if sample else casi


def predici(query: str, ticket_id: str) -> Dict[str, Any]:
    """
    Esegue il sistema reale su una richiesta e restituisce quanto serve agli
    scorer.

    MLflow invoca questa funzione per ogni riga del dataset, passandole i
    campi di `inputs`, e ne consegna il risultato agli scorer come `outputs`.

    I contesti sono restituiti come **elenco di testi separati** e non
    concatenati: `context_precision` valuta la pertinenza di ciascun chunk
    singolarmente e ne considera la posizione. Un unico blocco renderebbe la
    metrica cieca a quale chunk sia utile, che è proprio l'informazione
    necessaria per capire se il ranking funziona.

    Il leave-one-out è applicato qui, non a valle: senza, ogni ticket
    recupererebbe sé stesso e tutte le metriche di recupero risulterebbero
    perfette per costruzione.
    """
    esito = run_system(query, exclude_sources=[ticket_id])
    return {
        "answer": esito["answer"],
        # I ticket sono conservati integri nel payload: il testo valutato
        # deve essere lo stesso che il modello ha letto, quindi si usa il
        # medesimo formattatore impiegato per costruire il prompt.
        "contexts": [d.get("text", "") for d in esito["kb_docs"]]
                    + [ticket_as_context(t) for t in esito["kb_tickets"]],
        "escalated": esito["escalated"],
    }


# ==========================================================================
# Entrypoint
# ==========================================================================

def esegui(sample: int, con_scorer_mlflow: bool) -> Any:
    """
    Esegue la valutazione tramite `mlflow.genai.evaluate`.

    MLflow si occupa di: eseguire `predici` su ogni riga, tracciarne
    l'esecuzione, applicare gli scorer ai risultati e persistere il tutto come
    run. Non serve alcun ciclo esplicito sui casi né registrazione manuale
    delle metriche.

    Con `con_scorer_mlflow` si aggiungono due scorer nativi che misurano
    criteri analoghi con procedure diverse. Affiancarli non è ridondanza: se
    RAGAS e MLflow concordano, il giudizio è robusto rispetto
    all'implementazione; se divergono, il numero va preso con cautela, ed è
    un'informazione che una sola libreria non può dare.
    """
    import mlflow
    from mlflow.genai import evaluate

    from app import prompts

    scorers = costruisci_scorer()
    if con_scorer_mlflow:
        from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness

        scorers += [RetrievalGroundedness(), RelevanceToQuery()]

    casi = carica_casi(sample)
    logger.info("Valutazione su %d casi con %d scorer.", len(casi), len(scorers))

    prompts.register_agent_prompt()
    mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT}-eval")

    with mlflow.start_run(run_name="eval-ragas"):
        # Configurazione e versione dei prompt fra i parametri: senza, i
        # punteggi non sarebbero attribuibili a una configurazione precisa e
        # due run non sarebbero confrontabili a posteriori.
        mlflow.log_params(config.as_params())
        mlflow.log_params(prompts.as_params())
        mlflow.log_param("evaluation_framework", "ragas+mlflow")

        risultato = evaluate(data=casi, predict_fn=predici, scorers=scorers)

    return risultato


def main() -> int:
    """Entrypoint da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Valutazione RAG con metriche RAGAS eseguite tramite MLflow")
    parser.add_argument("--sample", type=int, default=20,
                        help="numero di ticket da valutare; ogni caso comporta "
                             "più chiamate al modello giudice. Il campionamento è "
                             "stratificato e garantisce almeno un caso per "
                             "sottocategoria, quindi non scende sotto 20")
    parser.add_argument("--no-mlflow-scorers", action="store_true",
                        help="usa solo le metriche RAGAS, senza gli scorer nativi MLflow")
    args = parser.parse_args()

    if not config.GEMINI_API_KEY:
        logger.error("Serve una chiave API: le metriche RAGAS sono calcolate da un LLM.")
        return 1

    risultato = esegui(args.sample, con_scorer_mlflow=not args.no_mlflow_scorers)

    print("\n=== Metriche aggregate ===")
    for nome, valore in sorted((risultato.metrics or {}).items()):
        if isinstance(valore, (int, float)):
            print(f"  {nome:48s} {valore:.4f}")
    print("\nDettaglio per caso e tracce disponibili nell'interfaccia MLflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
