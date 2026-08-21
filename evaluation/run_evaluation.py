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
from typing import Any, Dict, List, Set

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

def run_escalation_suite() -> Dict[str, float]:
    """
    Esegue il grafo su ogni caso e confronta la decisione con quella attesa.

    TODO(1) — Invocare il sistema senza passare dal nodo human_review:
        interessa la *decisione*, non l'interrupt. Due strade possibili:
        a) chiamare direttamente `escalation.decide()` dopo aver eseguito a
           mano retrieval e `generate_draft_answer` — veloce e isolato, ma
           salta il cablaggio del grafo;
        b) eseguire il grafo e leggere `escalation_triggers` dallo stato.
        Preferibile (b): misura il sistema com'è davvero in produzione.

    TODO(2) — Decidere se valutare anche i 135 ticket storici oltre ai casi
        scritti a mano. Sono ground truth vera, ma coprono solo i trigger
        deterministici §3 e in leave-one-out: da tenere come suite separata,
        non mescolata a questa.
    """
    cases = load_escalation_cases()
    results: List[EscalationCase] = []

    for c in cases:
        # TODO(1): sostituire con l'invocazione reale del sistema.
        predicted_escalate = False
        predicted_codes: List[str] = []

        results.append(EscalationCase(
            case_id=c["case_id"],
            predicted_escalate=predicted_escalate,
            expected_escalate=c["expected_escalate"],
            predicted_trigger_codes=predicted_codes,
            expected_trigger_codes=c["expected_trigger_codes"],
            trigger_family=c["trigger_family"],
        ))

    return evaluate_escalation(results)


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
        attese = set(rilevanza.get(subcat, {}).get("expected", []))
        casi_docs.append(RetrievalCase(
            query_id=tid, retrieved_ids=recuperati_d, relevant_ids=attese,
            top_score=risultati_d[0]["score"] if risultati_d else 0.0,
        ))

        per_caso.append({
            "ticket_id": tid,
            "subcategory": subcat,
            "tickets_retrieved": ", ".join(recuperati_t),
            "tickets_hit": bool(set(recuperati_t) & rilevanti_t),
            "tickets_top_score": round(risultati_t[0]["score"], 4) if risultati_t else 0.0,
            "docs_retrieved": ", ".join(recuperati_d),
            "docs_expected": ", ".join(sorted(attese)),
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

def run_answer_quality_suite() -> Dict[str, float]:
    """
    TODO(5) — Implementare con `mlflow.genai.evaluate()` e i suoi scorer
        (RetrievalGroundedness, RetrievalSufficiency, RelevanceToQuery,
        Correctness contro `resolution_summary`), più uno scorer custom di
        policy compliance. Richiede una chiave API attiva: è un giudizio dato
        da un modello, non una formula.
    """
    logger.warning("Suite 'answers' non ancora implementata (vedi TODO(5)).")
    return {}


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


def log_to_mlflow(metrics: Dict[str, float], suite: str) -> None:
    """
    Registra metriche e configurazione come run MLflow.

    Loggare `config.as_params()` insieme alle metriche è ciò che rende i run
    confrontabili: senza sapere con quali soglie e quale modello è stato
    prodotto un risultato, il numero da solo non dice nulla e non si possono
    confrontare due configurazioni.
    """
    try:
        import mlflow
        from app import config

        mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT}-eval")
        with mlflow.start_run(run_name=f"eval-{suite}"):
            mlflow.log_params(config.as_params())
            mlflow.log_metrics(metrics)
            logger.info("Metriche registrate su MLflow (suite=%s).", suite)
    except Exception:
        logger.exception("Logging su MLflow fallito: le metriche restano solo a video.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation del sistema di help desk")
    parser.add_argument("--suite", choices=["escalation", "retrieval", "answers", "all"],
                        default="all")
    parser.add_argument("--k", type=int, default=3, help="top-k per le metriche di retrieval")
    parser.add_argument("--sample", type=int, default=0,
                        help="valuta solo N ticket, campionati in modo stratificato (0 = tutti)")
    parser.add_argument("--no-mlflow", action="store_true", help="non registrare su MLflow")
    args = parser.parse_args()

    groups = []
    if args.suite in ("escalation", "all"):
        groups.append(run_escalation_suite())
    if args.suite in ("retrieval", "all"):
        groups.append(run_retrieval_suite(k=args.k, sample=args.sample))
    if args.suite in ("answers", "all"):
        groups.append(run_answer_quality_suite())

    metrics = flatten_metrics(*groups)

    print(f"\n=== Risultati (suite={args.suite}) ===")
    for name, value in sorted(metrics.items()):
        print(f"  {name:45s} {value:.4f}")

    if not args.no_mlflow and metrics:
        log_to_mlflow(metrics, args.suite)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
