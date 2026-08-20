"""
ingest.py
=========
Entrypoint del **job di ingestion**: popola le collection Qdrant a partire dai
file in `app/knowledge_base/`, poi termina.

    python -m app.ingest

In Docker Compose gira come servizio one-shot che deve completare con successo
prima che il backend accetti richieste (`depends_on: service_completed_successfully`).
Il motivo per cui è un servizio a sé e non codice eseguito all'avvio del
backend: se il backend scalasse a più repliche, ciascuna proverebbe a
indicizzare in parallelo la stessa collection. Con un job dedicato
l'ingestion avviene una volta sola, in un punto ben definito del ciclo di vita.

Comportamento (requisito di deploy): **se una collection non esiste viene
creata e popolata, altrimenti viene aggiornata** in modo incrementale — vedi
`sync_collection` in `app/retrieval.py`. Un riavvio senza modifiche alla
knowledge base non produce nessuna chiamata all'API di embedding.

Attesa di Qdrant
----------------
Il container di Qdrant può non essere ancora pronto quando questo job parte.
Invece di affidarsi solo all'ordine di avvio dichiarato nel compose, il job
riprova a connettersi con backoff: è più robusto di un healthcheck, perché
verifica proprio l'operazione che ci interessa (interrogare le collection) e
non un generico "la porta risponde".
"""
import logging
import sys
import time

from app import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

# Attesa massima complessiva per la disponibilità di Qdrant.
MAX_WAIT_SECONDS = 60
RETRY_DELAY_SECONDS = 2


def wait_for_qdrant() -> bool:
    """Riprova finché Qdrant non risponde, o finché non scade il tempo massimo."""
    from app.retrieval import _client

    target = config.QDRANT_URL or f"path={config.QDRANT_PATH}"
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    attempt = 0

    while time.monotonic() < deadline:
        attempt += 1
        try:
            _client.get_collections()
            logger.info("Qdrant raggiungibile (%s) al tentativo %d.", target, attempt)
            return True
        except Exception as exc:
            logger.info("Qdrant non ancora pronto (%s): %s", target, exc.__class__.__name__)
            time.sleep(RETRY_DELAY_SECONDS)

    logger.error("Qdrant non raggiungibile entro %d secondi (%s).", MAX_WAIT_SECONDS, target)
    return False


def main() -> int:
    logger.info("Avvio ingestion della knowledge base.")
    # Diagnostica esplicita: senza chiave l'ingestion non fallisce, ricade
    # sull'embedding di fallback e costruisce un indice di qualità scadente
    # senza dirlo. Meglio che si veda subito, nella prima riga di log.
    logger.info("Credenziali Gemini — %s", config.describe_credentials())

    if not wait_for_qdrant():
        return 1

    from app.retrieval import ensure_index

    try:
        stats = ensure_index()
    except Exception:
        logger.exception("Ingestion fallita.")
        return 1

    for name, s in stats.items():
        logger.info(
            "%s: %d punti (%s) — %d aggiornati o nuovi, %d invariati, %d rimossi",
            name, s["total"], "creata" if s["created"] else "aggiornata",
            s["upserted"], s["unchanged"], s["deleted"],
        )

    logger.info("Ingestion completata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
