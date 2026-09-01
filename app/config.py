"""
config.py
=========
Punto unico di configurazione del sistema: modelli, parametri di embedding e
soglie della logica di escalation.

Perché centralizzare: le soglie di escalation sono **iperparametri**, non
costanti. Nello step di evaluation con MLflow vorremo loggarle come parametri
di run e farne sweep per trovare il punto di lavoro migliore; averle sparse
nel codice lo renderebbe impossibile. Tutte sono sovrascrivibili da variabile
d'ambiente, così un container può essere avviato con soglie diverse senza
ricostruire l'immagine.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """
    Carica il file `.env` dalla root del progetto, se esiste.

    Perché serve: fuori da Docker nessuno popola l'ambiente al posto nostro.
    Prima di questa funzione occorreva esportare le variabili a mano, con un
    comando diverso a seconda della shell (`export ...` su bash, `set ...` sul
    prompt di Windows, `$env:...` su PowerShell) — una fonte di errori
    sproporzionata rispetto al problema, e con un sintomo insidioso: senza
    chiave il sistema non si ferma, ricade sui mock e sembra solo "di bassa
    qualità".

    Due regole:
    - **l'ambiente vince sul file**: una variabile già impostata non viene
      mai sovrascritta. In Docker Compose è `env_file`/`environment` a
      popolare l'ambiente, quindi qui il file non c'è nemmeno (è escluso dal
      contesto di build) e la funzione non fa nulla;
    - parsing minimale e senza dipendenze: `CHIAVE=valore`, commenti con `#`,
      virgolette rimosse, righe malformate ignorate in silenzio.
    """
    percorso = Path(__file__).resolve().parent.parent / ".env"
    if not percorso.is_file():
        return
    try:
        for riga in percorso.read_text(encoding="utf-8").splitlines():
            riga = riga.strip()
            if not riga or riga.startswith("#") or "=" not in riga:
                continue
            chiave, _, valore = riga.partition("=")
            chiave = chiave.strip()
            # Tollera la forma `export CHIAVE=valore`, comune nei .env
            # copiati da guide scritte per bash.
            if chiave.startswith("export "):
                chiave = chiave[len("export "):].strip()
            valore = valore.strip().strip('"').strip("'")
            if chiave and chiave not in os.environ:
                os.environ[chiave] = valore
    except OSError:
        # Un .env illeggibile non deve impedire l'avvio: le variabili
        # potrebbero comunque arrivare dall'ambiente.
        pass


_load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# --------------------------------------------------------------------------
# Provider LLM ed embedding (Google Gemini)
# --------------------------------------------------------------------------

# La chiave abilita le chiamate reali. Senza chiave il sistema resta
# pienamente funzionante in "modalità mock" (vedi llm.py e retrieval.py):
# scelta deliberata, permette di sviluppare, testare e far girare la demo
# senza credenziali e senza costi.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Nomi modello configurabili: la famiglia Gemini evolve rapidamente e non
# vogliamo dover toccare il codice per provarne uno diverso.
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_EMBEDDING_MODEL = _env_str("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Modello usato come giudice nell'evaluation delle risposte. Configurabile
# separatamente perché ha senso poterlo tenere diverso (e più capace) da
# quello che genera le risposte: un giudice che condivide con l'esaminato
# esattamente gli stessi punti ciechi tende a non vederne gli errori.
JUDGE_MODEL = _env_str("JUDGE_MODEL", GEMINI_MODEL)

# gemini-embedding-001 produce 3072 dimensioni di default, riducibili via
# `output_dimensionality` grazie al Matryoshka Representation Learning: i
# primi N valori restano da soli informativi. Con ~170 punti totali in indice,
# 768 è ampiamente sufficiente e riduce spazio e tempo di query.
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 768)

# Quanti testi mandare per chiamata in fase di indicizzazione.
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)

# Tentativi complessivi su un batch di embedding prima di arrendersi, con
# attesa che raddoppia fra l'uno e l'altro. Le quote del provider sono al
# minuto, quindi un errore di quota è quasi sempre transitorio: ritentare lo
# risolve, fallire farebbe perdere tutto il lavoro già svolto.
EMBEDDING_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 8)

# Dimensione dell'embedding di fallback (hashing trick) usato senza API key.
FALLBACK_EMBEDDING_DIM = 256


# --------------------------------------------------------------------------
# Qdrant
# --------------------------------------------------------------------------

# In sviluppo locale l'indice sta su disco (nessun server esterno). In Docker
# Compose si valorizza QDRANT_URL e lo stesso client punta al servizio
# containerizzato: cambia solo l'argomento del costruttore.
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PATH = _env_str("QDRANT_PATH", "qdrant_data")


# --------------------------------------------------------------------------
# Soglie della logica di escalation
# --------------------------------------------------------------------------

# POL-006 §4 primo criterio: "confidence score below 0.65".
# Il valore è preso letteralmente dalla policy, non scelto da noi.
CONFIDENCE_THRESHOLD = _env_float("ESCALATION_CONFIDENCE_THRESHOLD", 0.65)

# POL-006 §4 secondo criterio: "retrieval from both knowledge bases returns no
# result above the minimum similarity threshold configured for the system".
# La policy demanda esplicitamente il valore alla configurazione di sistema:
# è questo. VA RITARATO sui punteggi reali degli embedding Gemini — il valore
# attuale è un punto di partenza, non una misura.
MIN_RETRIEVAL_SCORE = _env_float("ESCALATION_MIN_RETRIEVAL_SCORE", 0.45)

# Segnale "precedente": tra i ticket storici recuperati sopra questa
# similarità, se la quota di quelli che furono escalati supera la soglia
# sotto, è un indizio che anche questo caso vada escalato.
PRECEDENT_SCORE_FLOOR = _env_float("ESCALATION_PRECEDENT_SCORE_FLOOR", 0.55)
PRECEDENT_ESCALATION_RATIO = _env_float("ESCALATION_PRECEDENT_RATIO", 0.5)

# Quanti risultati recuperare da ciascuna delle due knowledge base.
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 3)

# Indicizzare automaticamente all'import di app/retrieval.py.
# True in sviluppo locale (basta avviare uvicorn), False in Docker Compose:
# lì è un servizio di ingestion dedicato a popolare Qdrant prima che il
# backend accetti richieste, e due processi che indicizzano in parallelo
# sarebbero solo lavoro duplicato.
AUTO_INDEX = os.environ.get("AUTO_INDEX", "true").lower() not in ("0", "false", "no")


def active_embedding_dim() -> int:
    """Dimensione dei vettori effettivamente prodotti dal provider attivo."""
    return EMBEDDING_DIM if GEMINI_API_KEY else FALLBACK_EMBEDDING_DIM


def embedding_provider() -> str:
    return "gemini" if GEMINI_API_KEY else "hashing-fallback"


# --------------------------------------------------------------------------
# MLflow (tracing ed evaluation)
# --------------------------------------------------------------------------

# In Docker Compose punta al servizio containerizzato (es.
# "http://mlflow:5000"). Se non impostata, MLflow scrive su ./mlruns.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = _env_str("MLFLOW_EXPERIMENT", "helpdesk-agent")
MLFLOW_ENABLED = os.environ.get("MLFLOW_ENABLED", "true").lower() not in ("0", "false", "no")


def describe_backend() -> str:
    """
    Riporta a quale servizio Google punta il client: API diretta o Vertex AI.

    Sono due prodotti distinti, con **quote separate**, e la differenza si nota
    solo quando una richiesta viene rifiutata: un errore che cita
    `aiplatform.googleapis.com` viene da Vertex, uno che cita
    `generativelanguage.googleapis.com` dall'API diretta. Poiché le quote di
    embedding predefinite su Vertex sono più strette, sapere quale delle due si
    sta usando è il primo passo per interpretare un 429.

    La scelta non è esplicita nel nostro codice: la fa l'SDK in base
    all'ambiente. Basta `GOOGLE_GENAI_USE_VERTEXAI=true`, oppure la presenza di
    `GOOGLE_CLOUD_PROJECT` insieme alle credenziali applicative, perché il
    client passi a Vertex senza che nulla lo segnali.
    """
    if not GEMINI_API_KEY:
        return "nessun backend attivo (chiave assente)"
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        url = client._api_client._http_options.base_url
        nome = "Vertex AI" if client.vertexai else "API Gemini diretta"
        return f"{nome} ({url})"
    except Exception as e:
        return f"backend non determinabile: {type(e).__name__}"


def describe_credentials() -> str:
    """
    Descrizione della credenziale attiva, sicura da mandare a log.

    Mostra solo i primi e gli ultimi caratteri della chiave: basta a
    verificare che sia arrivata quella giusta, senza esporla. Serve perché
    l'assenza della chiave non produce un errore ma un *fallback silenzioso*
    a embedding e risposte finte — un guasto che si manifesta come "qualità
    inspiegabilmente bassa" invece che come eccezione.
    """
    key = GEMINI_API_KEY
    if not key:
        presenti = [n for n in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if n in os.environ]
        if presenti:
            return (f"NESSUNA CHIAVE ATTIVA: {', '.join(presenti)} è presente "
                    f"nell'ambiente ma vuota. Con Docker Compose ricorda che "
                    f"`environment:` sovrascrive `env_file:`.")
        return ("NESSUNA CHIAVE ATTIVA: né GEMINI_API_KEY né GOOGLE_API_KEY "
                "sono presenti nell'ambiente. In locale il file .env NON viene "
                "letto automaticamente: esporta le variabili a mano.")
    return f"chiave attiva: {key[:6]}...{key[-4:]} ({len(key)} caratteri)"


def as_params() -> dict:
    """
    Configurazione attiva in forma loggabile come parametri di un run MLflow.

    La chiave API è deliberatamente esclusa: un parametro MLflow finisce in
    chiaro nel tracking store e sarebbe una fuga di credenziali.
    """
    return {
        "llm_provider": "gemini" if GEMINI_API_KEY else "mock",
        "llm_model": GEMINI_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding_provider": embedding_provider(),
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "embedding_dim": active_embedding_dim(),
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_retrieval_score": MIN_RETRIEVAL_SCORE,
        "precedent_score_floor": PRECEDENT_SCORE_FLOOR,
        "precedent_escalation_ratio": PRECEDENT_ESCALATION_RATIO,
    }
