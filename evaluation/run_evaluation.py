"""
evaluation/run_evaluation.py
============================
Harness di evaluation: esegue il sistema sui dataset di test, calcola le
metriche di `metrics.py` e le registra come run MLflow.

    python -m evaluation.run_evaluation --suite escalation
    python -m evaluation.run_evaluation --suite retrieval
    python -m evaluation.run_evaluation --suite all

STATO: **abbozzo**. Lo scheletro, il caricamento dei dataset e il logging su
MLflow ci sono; i punti marcati `TODO` sono quelli che richiedono decisioni
ancora aperte o un sistema configurato con credenziali reali. Vedi TODO.md.

Perché eseguire l'evaluation solo dopo aver attivato Gemini e tarato le
soglie: con l'embedding di fallback i punteggi di similarità sono bassi e
compressi, quindi il trigger di grounding scatta quasi sempre e il sistema
sovra-escala. Misurare adesso produrrebbe numeri che descrivono la
configurazione provvisoria, non il sistema.
"""
import argparse
import json
import logging
from pathlib import Path
import random
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Set

from app import config
from app.retrieval import search_kb_docs, search_kb_tickets
from evaluation.metrics import (
    EscalationCase,
    RetrievalCase,
    aggregate_retrieval,
    evaluate_escalation,
    flatten_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluation")

DATASETS_DIR = Path(__file__).parent / "datasets"
KB_DIR = Path(__file__).parent.parent / "app" / "knowledge_base"


# ==========================================================================
# Caricamento dei dataset
# ==========================================================================

def load_escalation_cases() -> List[Dict[str, Any]]:
    """
    Casi scritti a mano per i criteri che lo storico non copre.

    Contiene sia positivi (devono escalare) sia negativi (devono essere
    risolti dall'agente): senza i negativi non si può misurare la precision,
    e un sistema che escala tutto otterrebbe recall perfetto.
    """
    path = DATASETS_DIR / "escalation_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    pos = sum(1 for c in cases if c["expected_escalate"])
    logger.info("Caricati %d casi di escalation (%d positivi, %d negativi) da %s",
                len(cases), pos, len(cases) - pos, path.name)
    return cases


def load_ticket_cases() -> List[Dict[str, Any]]:
    """
    Ground truth ricavata dallo storico: `was_escalated_to_human` è
    l'etichetta, `subject` + `description` sono la query.

    ATTENZIONE AL LEAKAGE: questi stessi ticket sono indicizzati in
    `kb_tickets`, quindi il retrieval troverebbe il ticket identico a sé
    stesso e la valutazione sarebbe gonfiata. Va usata la modalità
    leave-one-out — vedi `TODO` in `run_retrieval_suite`.
    """
    tickets = json.loads((KB_DIR / "past_tickets.json").read_text(encoding="utf-8"))
    return [
        {
            "case_id": t["ticket_id"],
            "query": f"{t['subject']}\n\n{t['description']}",
            "expected_escalate": t["was_escalated_to_human"],
            "category": t["category"],
            "subcategory": t["subcategory"],
        }
        for t in tickets
    ]


def load_policy_relevance() -> Dict[str, Dict[str, List[str]]]:
    """Mappa sottocategoria -> policy attese, ground truth per kb_docs."""
    data = json.loads((DATASETS_DIR / "policy_relevance.json").read_text(encoding="utf-8"))
    return data["by_subcategory"]


def stratified_sample(cases: List[Dict[str, Any]], n: int,
                      key: str = "subcategory", seed: int = 42) -> List[Dict[str, Any]]:
    """
    Sottocampione che preserva le proporzioni per `key`.

    Serve a iterare in fretta durante lo sviluppo: una run completa costa
    135 query, ciascuna con embedding e generazione. Un campionamento
    casuale semplice rischierebbe di lasciare fuori del tutto le
    sottocategorie meno numerose (le più piccole hanno 5 ticket su 135),
    proprio quelle su cui il sistema ha più probabilità di sbagliare.

    Il seed è fisso: due esecuzioni consecutive devono valutare lo stesso
    campione, altrimenti le differenze tra run sono rumore e non segnale.

    ATTENZIONE ALLA NUMEROSITÀ MINIMA. Ogni gruppo contribuisce con almeno un
    caso (`max(1, ...)`), quindi il campione non può scendere sotto il numero
    di gruppi: con 20 sottocategorie, chiedere 6 casi ne restituisce comunque
    20. È voluto — un campione che ignora del tutto le sottocategorie meno
    numerose non è rappresentativo — ma va conosciuto quando il costo per caso
    è alto, come nelle suite che impiegano un modello giudice.
    """
    if n <= 0 or n >= len(cases):
        return cases

    gruppi: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        gruppi.setdefault(c[key], []).append(c)

    rng = random.Random(seed)
    campione: List[Dict[str, Any]] = []
    for valore, gruppo in sorted(gruppi.items()):
        quota = max(1, round(n * len(gruppo) / len(cases)))
        campione.extend(rng.sample(gruppo, min(quota, len(gruppo))))

    rng.shuffle(campione)
    logger.info("Campione stratificato: %d casi su %d, %d gruppi rappresentati",
                len(campione), len(cases), len(gruppi))
    return campione


# ==========================================================================
# Suite: decisione di escalation
# ==========================================================================

def run_system(query: str, exclude_sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Esegue il sistema reale su una richiesta e ne restituisce l'esito.

    Invoca il grafo LangGraph completo, non le singole funzioni: misuriamo il
    sistema com'è in produzione, cablaggio compreso. Se la decisione è di
    escalare, il grafo si sospende su `interrupt()` e `invoke` ritorna con la
    chiave `__interrupt__`, il cui payload contiene bozza, confidenza e
    l'elenco completo dei trigger scattati. Se invece non escala, il grafo
    arriva a `finalize` e l'esito è nello stato.

    Ogni chiamata usa un `thread_id` nuovo: i casi di test sono indipendenti e
    non devono ereditare la cronologia l'uno dell'altro.
    """
    from app.graph import graph

    config_lg = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    stato = {"user_query": query, "exclude_sources": exclude_sources or []}
    out = graph.invoke(stato, config_lg)

    if "__interrupt__" in out:
        payload = out["__interrupt__"][0].value
        return {
            "escalated": True,
            "triggers": payload.get("triggers", []),
            "answer": payload.get("draft_answer", ""),
            "confidence": payload.get("confidence"),
            "kb_docs": out.get("kb_docs_context", []),
            "kb_tickets": out.get("kb_tickets_context", []),
        }

    return {
        "escalated": False,
        "triggers": [],
        "answer": out.get("final_answer", ""),
        "confidence": out.get("confidence"),
        "kb_docs": out.get("kb_docs_context", []),
        "kb_tickets": out.get("kb_tickets_context", []),
    }


def run_escalation_suite_b() -> Dict[str, float]:
    """
    SUITE B — casi scritti a mano (`escalation_cases.json`).

    È la suite che copre i criteri di POL-006 §4 e i redirect fuori scope,
    che lo storico dei ticket non contiene affatto: nessun ticket passato è
    stato escalato "perché l'AI non era sicura", visto che sono tutti stati
    gestiti da umani.

    Nessuna esclusione dal retrieval: queste query non sono ticket indicizzati.
    """
    cases = load_escalation_cases()
    risultati: List[EscalationCase] = []
    per_caso: List[Dict[str, Any]] = []

    for c in cases:
        esito = run_system(c["query"])
        codici = [t["code"] for t in esito["triggers"]]

        risultati.append(EscalationCase(
            case_id=c["case_id"],
            predicted_escalate=esito["escalated"],
            expected_escalate=c["expected_escalate"],
            predicted_trigger_codes=codici,
            expected_trigger_codes=c["expected_trigger_codes"],
            trigger_family=c["trigger_family"],
        ))
        per_caso.append({
            "case_id": c["case_id"],
            "query": c["query"][:90],
            "expected": c["expected_escalate"],
            "predicted": esito["escalated"],
            "corretto": esito["escalated"] == c["expected_escalate"],
            "expected_codes": ", ".join(c["expected_trigger_codes"]),
            "predicted_codes": ", ".join(codici),
            "confidence": esito["confidence"],
            "family": c["trigger_family"],
        })

    _log_table(per_caso, "escalation_suite_b_per_case")
    return {f"suiteB/{k}": v for k, v in evaluate_escalation(risultati).items()}


def run_escalation_suite_a(sample: int = 0) -> Dict[str, float]:
    """
    SUITE A — i ticket storici, con `was_escalated_to_human` come ground truth.

    Attenzione a cosa misura davvero: l'EDA aveva mostrato che queste
    etichette sono spiegate al 100% dai trigger deterministici di POL-006 §3.
    Le regole quindi sono note e fisse; quello che questa suite mette alla
    prova è se **il modello estrae correttamente i segnali** su cui le regole
    operano (categoria del ticket, presenza di approvazioni, impatto
    multi-utente). È l'anello debole del percorso mandatorio.

    Il leave-one-out è indispensabile qui, più ancora che nel retrieval: il
    payload dei ticket contiene l'esito dell'escalation, e senza esclusione il
    modello lo leggerebbe nel contesto.
    """
    cases = load_ticket_cases()
    if sample:
        cases = stratified_sample(cases, sample)

    risultati: List[EscalationCase] = []
    per_caso: List[Dict[str, Any]] = []

    for c in cases:
        esito = run_system(c["query"], exclude_sources=[c["case_id"]])
        codici = [t["code"] for t in esito["triggers"]]

        risultati.append(EscalationCase(
            case_id=c["case_id"],
            predicted_escalate=esito["escalated"],
            expected_escalate=c["expected_escalate"],
            predicted_trigger_codes=codici,
            # Lo storico non annota quale clausola ha fatto scattare
            # l'escalation, quindi non c'è un atteso per-trigger da
            # confrontare: il breakdown per trigger vive nella suite B.
            expected_trigger_codes=[],
            trigger_family="mandatory" if c["expected_escalate"] else "none",
        ))
        per_caso.append({
            "ticket_id": c["case_id"],
            "subcategory": c["subcategory"],
            "expected": c["expected_escalate"],
            "predicted": esito["escalated"],
            "corretto": esito["escalated"] == c["expected_escalate"],
            "predicted_codes": ", ".join(codici),
            "confidence": esito["confidence"],
        })

    _log_table(per_caso, "escalation_suite_a_per_case")

    metriche = {f"suiteA/{k}": v for k, v in evaluate_escalation(risultati).items()}

    # Accuratezza per sottocategoria: se il sistema sbaglia, serve sapere se
    # sbaglia dappertutto o su un'area specifica.
    per_sub: Dict[str, List[bool]] = {}
    for r, c in zip(risultati, cases):
        per_sub.setdefault(c["subcategory"], []).append(
            r.predicted_escalate == r.expected_escalate)
    for sub, esiti in sorted(per_sub.items()):
        chiave = sub.replace("/", "-").replace(" ", "_")
        metriche[f"suiteA/subcategory/{chiave}/accuracy"] = sum(esiti) / len(esiti)

    return metriche


# ==========================================================================
# Suite: retrieval
# ==========================================================================

def run_retrieval_suite(k: int = 3, sample: int = 0) -> Dict[str, float]:
    """
    Misura la qualità del retrieval sulle due collection, in leave-one-out.

    Ground truth:
    - **kb_tickets**: è rilevante un ticket della stessa sottocategoria della
      query. Proxy automatica, quindi senza costo di annotazione, ma con un
      effetto collaterale da conoscere: una sottocategoria ha fino a 10
      ticket, quindi i rilevanti in leave-one-out sono fino a 9 e `recall@3`
      non può superare 0,33 nemmeno per un sistema perfetto. Per questo le
      metriche primarie sono `hit_rate@k` e `mrr`, che non hanno quel tetto,
      e `capped_recall@k` normalizza sul massimo ottenibile.
    - **kb_docs**: sono rilevanti le policy elencate come `expected` in
      `policy_relevance.json`, mappa scritta a mano a livello di documento.

    Il leave-one-out si applica solo a kb_tickets: le policy non sono mai
    usate come query, quindi non possono recuperare sé stesse.
    """
    cases = load_ticket_cases()
    if sample:
        cases = stratified_sample(cases, sample)

    rilevanza = load_policy_relevance()
    casi_tickets: List[RetrievalCase] = []
    casi_docs: List[RetrievalCase] = []
    per_caso: List[Dict[str, Any]] = []

    # Indice sottocategoria -> ticket, per costruire l'insieme dei rilevanti.
    tutti = load_ticket_cases()
    per_subcat: Dict[str, Set[str]] = {}
    for t in tutti:
        per_subcat.setdefault(t["subcategory"], set()).add(t["case_id"])

    for c in cases:
        tid, query, subcat = c["case_id"], c["query"], c["subcategory"]

        # --- kb_tickets, in leave-one-out ---
        risultati_t = search_kb_tickets(query, k=k, exclude_sources=[tid])
        recuperati_t = [r["source"] for r in risultati_t]
        rilevanti_t = per_subcat.get(subcat, set()) - {tid}
        casi_tickets.append(RetrievalCase(
            query_id=tid, retrieved_ids=recuperati_t, relevant_ids=rilevanti_t,
            top_score=risultati_t[0]["score"] if risultati_t else 0.0,
        ))

        # --- kb_docs ---
        risultati_d = search_kb_docs(query, k=k)
        # Il payload porta il policy_id (POL-00x): la rilevanza è a livello di
        # documento, non di singola sezione, quindi si deduplica mantenendo
        # l'ordine di ranking, che è ciò su cui l'MRR si basa.
        recuperati_d: List[str] = []
        for r in risultati_d:
            pid = r.get("policy_id")
            if pid and pid not in recuperati_d:
                recuperati_d.append(pid)
        mappa = rilevanza.get(subcat, {})
        attese = set(mappa.get("expected", []))
        # Le policy "acceptable" sono pertinenti ma non indispensabili: non
        # contano come successo nel recall, ma non devono nemmeno essere
        # contate come errore nella precision (vedi policy_relevance.json).
        tollerate = set(mappa.get("acceptable", []))
        casi_docs.append(RetrievalCase(
            query_id=tid, retrieved_ids=recuperati_d, relevant_ids=attese,
            top_score=risultati_d[0]["score"] if risultati_d else 0.0,
            ignored_ids=tollerate,
        ))

        per_caso.append({
            "ticket_id": tid,
            "subcategory": subcat,
            "tickets_retrieved": ", ".join(recuperati_t),
            "tickets_hit": bool(set(recuperati_t) & rilevanti_t),
            "tickets_top_score": round(risultati_t[0]["score"], 4) if risultati_t else 0.0,
            "docs_retrieved": ", ".join(recuperati_d),
            "docs_expected": ", ".join(sorted(attese)),
            "docs_acceptable": ", ".join(sorted(tollerate)),
            "docs_hit": bool(set(recuperati_d) & attese),
            "docs_top_score": round(risultati_d[0]["score"], 4) if risultati_d else 0.0,
        })

    metriche = {}
    for prefisso, gruppo in (("kb_tickets", casi_tickets), ("kb_docs", casi_docs)):
        for nome, valore in aggregate_retrieval(gruppo, k=k).items():
            metriche[f"retrieval/{prefisso}/{nome}"] = valore

    _log_table(per_caso, "retrieval_per_case")
    return metriche


# ==========================================================================
# Suite: qualità delle risposte
# ==========================================================================

def run_answer_quality_suite(sample: int = 20) -> Dict[str, float]:
    """
    Qualità delle risposte, giudicata da un LLM (vedi `evaluation/judge.py`).

    Campionata di default: ogni caso costa una generazione più una chiamata di
    giudizio, quindi valutare tutti i 135 ticket a ogni iterazione non è
    sostenibile né necessario per capire se la qualità sta salendo o scendendo.

    Vengono giudicate **anche le risposte dei ticket escalati**: la bozza non
    è scartata quando si escala, viene consegnata all'operatore che la corregge.
    Una bozza scadente gli fa perdere tempo anche quando la decisione di
    escalare era giusta.
    """
    from evaluation.judge import aggrega, giudica

    if not config.GEMINI_API_KEY:
        logger.warning("Suite 'answers' saltata: richiede una chiave API attiva.")
        return {}

    cases = stratified_sample(load_ticket_cases(), sample) if sample else load_ticket_cases()
    giudizi = []
    per_caso: List[Dict[str, Any]] = []

    for c in cases:
        esito = run_system(c["query"], exclude_sources=[c["case_id"]])
        g = giudica(c["case_id"], c["query"], esito["answer"],
                    esito["kb_docs"], esito["kb_tickets"])
        if g is None:
            continue
        giudizi.append(g)
        per_caso.append({
            "ticket_id": c["case_id"],
            "subcategory": c["subcategory"],
            "escalated": esito["escalated"],
            "groundedness": g.groundedness,
            "relevance": g.relevance,
            "policy_compliance": g.policy_compliance,
            "policy_violation": g.policy_violation or "",
            "reasoning": g.reasoning,
            "answer": esito["answer"][:200],
        })

    _log_table(per_caso, "answers_per_case")
    return aggrega(giudizi, totale_casi=len(cases))


# ==========================================================================
# Entrypoint
# ==========================================================================

def _log_table(rows: List[Dict[str, Any]], name: str) -> None:
    """
    Salva i risultati per singolo caso come artifact MLflow.

    Un `hit_rate = 0,71` non è azionabile: dice che qualcosa non va, non
    cosa. La tabella per-caso è ciò che permette di aprire i fallimenti e
    capire se sono concentrati su una sottocategoria, su punteggi bassi o su
    una policy mappata male. Con 135 casi si legge a mano.
    """
    if not rows:
        return
    try:
        import mlflow
        mlflow.log_table(data={k: [r[k] for r in rows] for k in rows[0]},
                         artifact_file=f"{name}.json")
        logger.info("Tabella per-caso '%s' salvata come artifact (%d righe).", name, len(rows))
    except Exception:
        # Fallback su file locale: i dettagli per-caso sono troppo utili per
        # perderli solo perché MLflow non è raggiungibile.
        out = Path(f"{name}.json")
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.warning("MLflow non disponibile: tabella scritta in %s", out)


@contextmanager
def mlflow_run(suite: str, enabled: bool):
    """
    Apre il run MLflow **prima** dell'esecuzione delle suite.

    L'ordine non è un dettaglio. `mlflow.log_table`, usato per salvare il
    dettaglio per caso, richiede un run attivo e ne apre uno implicitamente se
    non lo trova: aprendo il run solo alla fine, per registrare le metriche, le
    tabelle sarebbero finite in un run diverso da quello delle metriche che
    devono spiegare — e il secondo `start_run` avrebbe comunque fallito,
    trovandone già uno attivo.

    Registra come parametri sia la configurazione sia la versione dei prompt:
    è ciò che rende confrontabili due valutazioni a distanza di tempo.
    """
    if not enabled:
        yield
        return

    try:
        import mlflow

        from app import prompts
        from evaluation.judge import JUDGE_PROMPT

        # Registrati prima di aprire il run: le metriche di una valutazione
        # vanno attribuite tanto al prompt dell'agente quanto a quello del
        # giudice, perché una modifica del secondo sposta i punteggi senza che
        # il sistema valutato sia cambiato.
        prompts.register_agent_prompt()
        prompts.ensure_registered(prompts.JUDGE_PROMPT_NAME, JUDGE_PROMPT,
                                  commit_message="Prompt del giudice di valutazione")

        mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT}-eval")
        with mlflow.start_run(run_name=f"eval-{suite}"):
            mlflow.log_params(config.as_params())
            mlflow.log_params(prompts.as_params())
            yield
            logger.info("Run di valutazione registrato su MLflow (suite=%s).", suite)

    except Exception:
        logger.exception("Registrazione su MLflow fallita: le metriche restano solo a video.")
        yield


def log_metrics(metrics: Dict[str, float]) -> None:
    """Registra le metriche nel run già aperto da `mlflow_run`."""
    if not metrics:
        return
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metrics(metrics)
    except Exception:
        logger.warning("Impossibile registrare le metriche.", exc_info=True)


def main() -> int:
    """
    Punto d'ingresso da riga di comando.

    Le suite girano **dentro** il contesto del run MLflow, non prima: le
    tabelle per caso vengono scritte durante l'esecuzione e devono finire nel
    medesimo run delle metriche che spiegano. Aprire il run solo alla fine le
    disperderebbe in un run separato.

    Le metriche vengono comunque stampate a video anche quando la
    registrazione fallisce: uno strumento di misura che non produce un
    risultato leggibile perché è indisponibile la telemetria sarebbe inutile
    proprio nelle situazioni in cui serve di più.
    """
    parser = argparse.ArgumentParser(description="Evaluation del sistema di help desk")
    parser.add_argument(
        "--suite",
        choices=["retrieval", "escalation-a", "escalation-b", "escalation", "answers", "all"],
        default="all",
        help="retrieval | escalation-a (ticket storici) | escalation-b (casi scritti a mano) "
             "| escalation (entrambe) | answers | all",
    )
    parser.add_argument("--k", type=int, default=3, help="top-k per le metriche di retrieval")
    parser.add_argument("--sample", type=int, default=0,
                        help="valuta solo N ticket, campionati in modo stratificato (0 = tutti)")
    parser.add_argument("--judge-sample", type=int, default=20,
                        help="quanti casi far giudicare dall'LLM nella suite 'answers'")
    parser.add_argument("--no-mlflow", action="store_true", help="non registrare su MLflow")
    args = parser.parse_args()

    groups = []
    with mlflow_run(args.suite, enabled=not args.no_mlflow):
        if args.suite in ("retrieval", "all"):
            groups.append(run_retrieval_suite(k=args.k, sample=args.sample))
        if args.suite in ("escalation-a", "escalation", "all"):
            groups.append(run_escalation_suite_a(sample=args.sample))
        if args.suite in ("escalation-b", "escalation", "all"):
            groups.append(run_escalation_suite_b())
        if args.suite in ("answers", "all"):
            groups.append(run_answer_quality_suite(sample=args.judge_sample))

        metrics = flatten_metrics(*groups)
        log_metrics(metrics)

    print(f"\n=== Risultati (suite={args.suite}) ===")
    for name, value in sorted(metrics.items()):
        print(f"  {name:45s} {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
