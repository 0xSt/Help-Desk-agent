# Immagine del servizio MLFLOW (tracking server: tracing ed evaluation).
#
# Costruita in casa invece di usare un'immagine pubblica pronta, per due
# motivi: la versione di MLflow resta allineata a quella dichiarata nel
# pyproject del progetto (server e client che divergono causano errori di
# schema difficili da diagnosticare), e l'immagine include solo ciò che serve.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /mlflow

# Il server ha bisogno di mlflow e del driver per il backend store.
# psycopg/altri driver non servono: qui il backend store è SQLite su volume.
RUN uv pip install --system --no-cache "mlflow>=3.0"

RUN useradd --create-home --uid 1000 mlflowuser \
    && mkdir -p /mlflow/artifacts /mlflow/db \
    && chown -R mlflowuser:mlflowuser /mlflow

USER mlflowuser

EXPOSE 5000

# --backend-store-uri: metadati (esperimenti, run, trace) su SQLite, in un
#   volume, così sopravvivono al riavvio del container.
# --artifacts-destination: artifact su volume separato.
# --host 0.0.0.0: altrimenti il server sarebbe raggiungibile solo da dentro
#   il container e non dagli altri servizi del compose.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/db/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
