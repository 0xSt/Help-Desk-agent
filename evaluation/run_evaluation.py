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
from typing import Any, Dict, List

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

def run_retrieval_suite(k: int = 3) -> Dict[str, float]:
    """
    Misura la qualità del retrieval in leave-one-out sui ticket storici.

    TODO(3) — Escludere il ticket stesso dai risultati. Qdrant supporta un
        filtro `must_not` sul payload: va passato un filtro su `source` ==
        ticket_id corrente. Serve quindi una variante di `search_kb_tickets`
        che accetti un filtro, oppure recuperare k+1 risultati e scartare
        l'auto-match a posteriori (più semplice, leggermente meno pulito).

    TODO(4) — Definire la ground truth di rilevanza:
        - `kb_tickets`: proxy ragionevole = stessa `subcategory` del ticket
          query. Approssimazione, ma automatica e senza annotazione manuale.
        - `kb_docs`: serve una mappatura categoria -> policy attese, scritta a
          mano una volta sola (es. Security -> POL-005, VPN -> POL-007).
          Senza, le metriche su kb_docs non sono calcolabili.
    """
    cases = load_ticket_cases()
    results: List[RetrievalCase] = []

    for c in cases:
        # TODO(3)/(4): eseguire la query escludendo sé stessa e costruire
        # l'insieme dei rilevanti.
        retrieved_ids: List[str] = []
        relevant_ids: set = set()
        top_score = 0.0

        results.append(RetrievalCase(
            query_id=c["case_id"],
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            top_score=top_score,
        ))

    return aggregate_retrieval(results, k=k)


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
    parser.add_argument("--no-mlflow", action="store_true", help="non registrare su MLflow")
    args = parser.parse_args()

    groups = []
    if args.suite in ("escalation", "all"):
        groups.append(run_escalation_suite())
    if args.suite in ("retrieval", "all"):
        groups.append(run_retrieval_suite(k=args.k))
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
