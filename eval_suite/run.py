"""
eval_suite/run.py
=================
Entrypoint della valutazione.

    python -m eval_suite.run --suite escalation
    python -m eval_suite.run --suite quality --sample 20
    python -m eval_suite.run --suite all --sample 20

Due suite, separate perché rispondono a domande diverse e si guastano in modo
indipendente.

**escalation** — la decisione di coinvolgere un operatore è corretta? Misurata
con regole deterministiche sui 43 casi etichettati, senza alcun giudice: la
verità è annotata, quindi introdurre un modello per stabilirla aggiungerebbe
soltanto rumore e costo.

**quality** — il contesto recuperato è pertinente e la risposta vi si attiene?
Misurata con i tre scorer nativi di MLflow, che sono giudizi di un modello
perché nessuna di queste proprietà si calcola con una formula chiusa.

| Scorer | Cosa misura | Su cosa opera |
|---|---|---|
| `RetrievalRelevance` | i documenti recuperati sono pertinenti alla richiesta | span `RETRIEVER` della traccia |
| `RetrievalGroundedness` | la risposta è sostenuta dal contesto recuperato | span `RETRIEVER` + risposta |
| `RelevanceToQuery` | la risposta affronta la domanda posta | `inputs` e `outputs` |

I primi due sono complementari e vanno letti insieme: fondatezza bassa con
pertinenza alta indica che il modello inventa pur avendo il materiale giusto;
entrambe basse indicano che il problema è a monte, nel recupero.

Riproducibilità
---------------
Ogni esecuzione registra come parametri del run la configurazione attiva, la
**versione del prompt** dell'agente con il relativo URI nel registry, il
modello impiegato come giudice e la composizione del dataset. È ciò che rende
confrontabili due valutazioni a distanza di tempo: un insieme di metriche senza
i parametri che l'hanno prodotto non è interpretabile, e non permette di
attribuire una differenza al prompt piuttosto che alle soglie o al modello.
"""
import argparse
import logging
import os
from typing import Any, Dict, List

import mlflow
from mlflow.entities import Feedback
from mlflow.genai import evaluate
from mlflow.genai.scorers import (
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    scorer,
)

from app import config, prompts
from eval_suite import datasets
from eval_suite.metrics import Esito, aggrega
from eval_suite import pipeline
from eval_suite.pipeline import predici

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eval")


def limita_concorrenza(workers: int) -> None:
    """
    Limita quanti casi MLflow valuta in parallelo.

    MLflow ne esegue **dieci alla volta** per impostazione predefinita. Ogni
    caso comporta due chiamate di embedding, una generazione e tre giudizi:
    dieci in parallelo saturano le quote al minuto del provider, che rispondono
    con 429 e fanno degradare il recupero a contesto vuoto — producendo metriche
    calcolate su nulla anziché un errore visibile.

    Il valore predefinito è basso di proposito. La valutazione non è un
    percorso interattivo: qualche minuto in più non ha costo, mentre una
    misurazione falsata da errori di quota va rifatta da capo.
    """
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(workers)
    logger.info("Concorrenza della valutazione limitata a %d casi paralleli.", workers)


def modello_giudice() -> str:
    """
    Identificativo del modello usato come giudice dagli scorer MLflow.

    MLflow, in assenza di indicazioni, userebbe un modello OpenAI: qui si passa
    esplicitamente il provider Google, nella forma `gemini:/<modello>` che
    MLflow instrada tramite LiteLLM. Si riusa `JUDGE_MODEL`, configurabile
    separatamente dal modello che genera le risposte, perché un giudice che
    condivide con il valutato gli stessi punti ciechi tende a non vederne gli
    errori.
    """
    return f"gemini:/{config.JUDGE_MODEL}"


@scorer
def escalation_corretta(outputs, expectations) -> Feedback:
    """
    Verifica per singolo caso se la decisione coincide con quella attesa.

    Restituisce un `Feedback` e non un numero perché porta con sé la
    motivazione: nell'interfaccia si vede subito se un caso è sbagliato per
    mancata escalation o per escalation superflua, senza aprire la traccia.

    Le metriche aggregate — richiamo, precisione, F2, MCC — non si calcolano
    qui: sono proprietà dell'insieme, non della riga, e vengono derivate dopo
    l'esecuzione in `_metriche_escalation`.
    """
    atteso, predetto = expectations["escalate"], outputs["escalated"]
    if atteso == predetto:
        esito = "escalation corretta" if atteso else "risolto correttamente dall'agente"
    else:
        esito = "MANCATA escalation" if atteso else "escalation superflua"
    return Feedback(name="escalation_corretta", value=atteso == predetto, rationale=esito)


def _metriche_escalation(risultato: Any, casi: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Deriva le metriche aggregate dal dettaglio per riga prodotto da MLflow.

    Il dataframe dei risultati conserva l'ordine delle righe del dataset, il
    che permette di riallineare ciascun esito al caso di partenza e recuperare
    la famiglia di segnale e le clausole attese, che non transitano dagli
    scorer.
    """
    df = risultato.result_df
    esiti: List[Esito] = []
    for caso, (_, riga) in zip(casi, df.iterrows()):
        risposta = riga.get("response") or {}
        if not isinstance(risposta, dict):
            logger.warning("Riga senza esito utilizzabile per %s: esclusa.", caso["case_id"])
            continue
        esiti.append(Esito(
            case_id=caso["case_id"],
            predetto=bool(risposta.get("escalated")),
            atteso=bool(caso["expectations"]["escalate"]),
            codici_predetti=list(risposta.get("triggers") or []),
            codici_attesi=list(caso["expectations"]["trigger_codes"]),
            famiglia=caso["trigger_family"],
        ))
    return aggrega(esiti)


def _parametri(nome_dataset: str, n_casi: int) -> Dict[str, Any]:
    """Configurazione e provenienza, registrate insieme alle metriche."""
    prompts.register_agent_prompt()
    params: Dict[str, Any] = {
        **config.as_params(),
        **prompts.as_params(),
        "eval/dataset": nome_dataset,
        "eval/n_cases": n_casi,
        "eval/judge_model": modello_giudice(),
        "eval/max_workers": os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS", "10"),
    }
    return params


def suite_escalation() -> Dict[str, float]:
    """Valuta la decisione di escalation sui casi etichettati."""
    casi = datasets.escalation_cases()
    logger.info("Suite escalation: %d casi.", len(casi))
    pipeline.inizia(len(casi))

    with mlflow.start_run(run_name="eval-escalation"):
        mlflow.log_params(_parametri("escalation_cases", len(casi)))
        risultato = evaluate(data=casi, predict_fn=predici, scorers=[escalation_corretta])
        metriche = _metriche_escalation(risultato, casi)
        mlflow.log_metrics(metriche)
    return metriche


def suite_quality(sample: int) -> Dict[str, float]:
    """Valuta pertinenza del contesto, fondatezza e aderenza della risposta."""
    casi = datasets.retrieval_cases(sample)
    logger.info("Suite quality: %d casi, giudice %s.", len(casi), modello_giudice())
    pipeline.inizia(len(casi))

    giudice = modello_giudice()
    scorers = [
        RetrievalRelevance(model=giudice),
        RetrievalGroundedness(model=giudice),
        RelevanceToQuery(model=giudice),
    ]

    with mlflow.start_run(run_name="eval-quality"):
        mlflow.log_params(_parametri("past_tickets", len(casi)))
        risultato = evaluate(data=casi, predict_fn=predici, scorers=scorers)
        metriche = {k: float(v) for k, v in (risultato.metrics or {}).items()
                    if isinstance(v, (int, float))}
    return metriche


def main() -> int:
    """Esegue le suite richieste e stampa le metriche."""
    parser = argparse.ArgumentParser(description="Valutazione del sistema di help desk")
    parser.add_argument("--suite", choices=["escalation", "quality", "all"], default="all")
    parser.add_argument("--workers", type=int, default=2,
                        help="casi valutati in parallelo. Valori alti saturano "
                             "le quote al minuto del provider e producono errori 429")
    parser.add_argument("--sample", type=int, default=20,
                        help="ticket da valutare nella suite quality; ogni caso "
                             "comporta più chiamate al modello giudice")
    args = parser.parse_args()

    limita_concorrenza(args.workers)
    # Quale servizio Google si sta usando: ha quote diverse, ed è la prima cosa
    # da sapere per interpretare un eventuale errore 429.
    logger.info("Backend Google — %s", config.describe_backend())
    mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT}-eval")
    metriche: Dict[str, float] = {}

    if args.suite in ("escalation", "all"):
        metriche.update(suite_escalation())
    if args.suite in ("quality", "all"):
        if not config.GEMINI_API_KEY:
            logger.error("La suite quality richiede una chiave API: le tre metriche "
                         "sono giudizi di un modello. Suite saltata.")
        else:
            metriche.update(suite_quality(args.sample))

    print("\n=== Metriche ===")
    for nome, valore in sorted(metriche.items()):
        print(f"  {nome:44s} {valore:.4f}")
    print("\nDettaglio per caso e tracce nell'interfaccia MLflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
