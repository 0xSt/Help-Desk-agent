"""
tracing.py
==========
Tracing del workflow LangGraph su **MLflow**.

Cosa traccia e perché
---------------------
Ogni chiamata a `/api/chat` esegue il grafo: retrieval sulle due KB,
generazione della bozza, decisione di escalation. Il tracing registra questa
esecuzione come una *trace* con uno span per nodo, e permette di rispondere a
domande che dai log non si ricavano: quali passaggi di policy sono stati
recuperati per quel ticket, con che punteggio, quale confidenza ha dichiarato
il modello, quale trigger ha causato l'escalation.

È il presupposto dello step di evaluation: le stesse trace prodotte in
esecuzione sono ciò su cui si calcolano poi le metriche.

Scelte di progetto
------------------
**MLflow è un servizio a sé.** L'URI del tracking server arriva da
`MLFLOW_TRACKING_URI` (in Docker Compose punterà al container `mlflow`). Senza
quella variabile MLflow scriverebbe in una cartella `mlruns/` locale: comodo
in sviluppo, ma va reso esplicito, non subito per caso.

**Il tracing non deve mai far cadere una richiesta.** Se il server MLflow è
irraggiungibile o l'autolog fallisce, il sistema continua a funzionare senza
tracciare: un help desk che smette di rispondere perché è giù la telemetria
sarebbe un pessimo scambio. Per lo stesso motivo l'inizializzazione è
idempotente e viene eseguita una sola volta all'avvio del processo.

**Autolog di LangChain.** LangGraph è costruito su LangChain, quindi
`mlflow.langchain.autolog()` intercetta l'esecuzione del grafo e produce
automaticamente uno span per nodo, senza dover strumentare i singoli nodi a
mano.
"""
import logging
from typing import Any, Dict, Optional

from app import config

logger = logging.getLogger(__name__)

_initialized = False


def setup_tracing() -> bool:
    """
    Configura il tracing MLflow. Ritorna True se è attivo.

    Idempotente: chiamarla più volte non produce doppie registrazioni.
    """
    global _initialized
    if _initialized:
        return True

    if not config.MLFLOW_ENABLED:
        logger.info("Tracing MLflow disabilitato da configurazione (MLFLOW_ENABLED).")
        return False

    try:
        import mlflow

        if config.MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            target = config.MLFLOW_TRACKING_URI
        else:
            # Nessun server configurato: MLflow scrive su ./mlruns.
            target = "store locale ./mlruns"

        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

        # Traccia automaticamente l'esecuzione del grafo LangGraph, uno span
        # per nodo. `log_traces=True` è ciò che ci interessa: le trace di
        # esecuzione, non il versionamento del grafo come modello.
        mlflow.langchain.autolog(log_traces=True)

        _initialized = True
        logger.info(
            "Tracing MLflow attivo — esperimento '%s' su %s",
            config.MLFLOW_EXPERIMENT, target,
        )
        return True

    except Exception:
        logger.exception(
            "Impossibile inizializzare il tracing MLflow: l'applicazione prosegue senza tracing."
        )
        return False


def log_run_configuration() -> None:
    """
    Registra la configurazione attiva (modelli, soglie, top-k) come parametri
    di un run MLflow dedicato all'avvio del servizio.

    Serve a rendere ricostruibile *con quale configurazione* è stato prodotto
    un certo insieme di trace: senza, confrontare due sessioni con soglie
    diverse diventa impossibile a posteriori.
    """
    if not _initialized:
        return
    try:
        import mlflow

        with mlflow.start_run(run_name="service-startup"):
            mlflow.log_params(config.as_params())
    except Exception:
        logger.warning("Impossibile registrare la configurazione su MLflow.", exc_info=True)


def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Decoratore per aggiungere uno span esplicito a una funzione.

    L'autolog copre già i nodi del grafo; questo serve per le funzioni che
    stanno *sotto* di essi e che vogliamo vedere separate — in particolare le
    due query di retrieval, i cui punteggi di similarità sono il segnale su
    cui si basa metà della logica di escalation.

    Se MLflow non è disponibile ritorna la funzione invariata, così il codice
    decorato resta eseguibile senza la dipendenza.
    """
    def decorator(fn):
        try:
            import mlflow

            return mlflow.trace(name=name, attributes=attributes)(fn)
        except Exception:
            return fn

    return decorator
