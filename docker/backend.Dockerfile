# Immagine del BACKEND (FastAPI + LangGraph) e del job di INGESTION: sono lo
# stesso codice, cambia solo il comando di avvio, quindi condividono l'immagine.
#
# Build multi-stage con uv: nello stage `builder` si risolve e installa
# l'ambiente virtuale, nell'immagine finale si copia solo quello. Il risultato
# non contiene uv né la cache di build.

FROM python:3.12-slim AS builder

# Binario di uv preso dall'immagine ufficiale: evita di installarlo con pip
# dentro l'immagine e di doverne gestire la versione a mano.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Prima solo i file di dipendenze: finché non cambiano, Docker riusa il layer
# con l'ambiente già installato e la build resta veloce anche modificando il
# codice applicativo.
COPY pyproject.toml uv.lock ./

# --frozen: fallisce se il lockfile non è allineato a pyproject.toml, invece
# di risolvere silenziosamente versioni diverse da quelle testate.
# --no-dev: esclude le dipendenze di sviluppo dall'immagine di runtime.
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim AS runtime

# Utente non privilegiato: il processo non ha motivo di girare come root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --chown=appuser:appuser app/ ./app/
# L'evaluation non serve a runtime, ma includerla permette di lanciarla nei
# container con `docker compose run`, evitando di dover replicare l'ambiente
# Python sull'host. Sono solo file di testo e JSON: peso trascurabile.
COPY --chown=appuser:appuser evaluation/ ./evaluation/

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
