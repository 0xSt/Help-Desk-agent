"""
retrieval.py
============
Retrieval reale sulle due knowledge base, usando Qdrant in modalità
**locale** (nessun server esterno: `QdrantClient(path=...)` scrive un
indice su disco nella directory del processo). Corrisponde allo "step
successivo" annunciato quando i nodi di retrieval in graph.py erano ancora
stub.

Due collection:
- `kb_docs`    — le policy IT in app/knowledge_base/policies/*.md, spezzate
                 per sezione (## Titolo).
- `kb_tickets` — i ticket storici in app/knowledge_base/past_tickets.json,
                 un punto per ticket.

INDICIZZAZIONE AUTOMATICA: al primo avvio del processo (import di questo
modulo), se le collection non esistono ancora, vengono create e popolate
leggendo i file bundled nel progetto. Le esecuzioni successive riusano
l'indice già su disco (`QDRANT_PATH`, di default "qdrant_data/" nella
working directory) — nessuna re-indicizzazione ad ogni avvio.

EMBEDDING: qui uso un embedding "hashing trick" deterministico, puro
Python, senza dipendenze pesanti né chiavi API — stesso spirito della
modalità mock di llm.py: fa funzionare l'intera pipeline (chunking,
indicizzazione, similarità coseno, ranking) subito e senza costi, ma la
qualità semantica è modesta (cattura soprattutto sovrapposizione lessicale,
non vera similarità di significato). È il punto di innesto naturale per un
provider di embedding reale (es. Voyage AI, che Anthropic raccomanda per
l'uso con Claude, o un modello locale tipo sentence-transformers): basta
sostituire `embed_text()`, ricreare le collection con la nuova dimensione
del vettore, e ri-eseguire l'indicizzazione.
"""
import hashlib
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent / "knowledge_base"
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_data")

EMBEDDING_DIM = 256
KB_DOCS_COLLECTION = "kb_docs"
KB_TICKETS_COLLECTION = "kb_tickets"

# Namespace fisso per generare ID punto deterministici (uuid5): ri-eseguire
# l'indicizzazione sullo stesso contenuto produce gli stessi ID, quindi
# l'upsert aggiorna i punti esistenti invece di duplicarli.
_ID_NAMESPACE = uuid.UUID("a13a1b2c-6f6e-4c1a-9c1f-0e5b6f3a2b10")

_client = QdrantClient(path=QDRANT_PATH)


# --------------------------------------------------------------------------
# Embedding "hashing trick" — vedi nota nel docstring del modulo.
# --------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", text.lower())


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Embedding deterministico via feature hashing: ogni token finisce in uno
    di `dim` bucket (indice = hash(token) % dim), con segno anch'esso
    derivato dall'hash. Il vettore risultante è normalizzato L2, così la
    similarità coseno in Qdrant si comporta in modo sensato.
    """
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        h = int(digest, 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# --------------------------------------------------------------------------
# Indicizzazione — kb_docs (policy Markdown)
# --------------------------------------------------------------------------

def _split_policy_into_sections(md_text: str, source: str) -> List[Dict[str, Any]]:
    """
    Spezza un file Markdown di policy in una sezione per ogni titolo `## `.
    Ogni sezione porta con sé il titolo del documento (riga `# Titolo`) come
    contesto, così un chunk isolato resta comprensibile fuori dal documento
    originale (es. "POL-001 — Sezione 5: Account Lockout Procedure: ...").
    """
    lines = md_text.splitlines()
    doc_title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else source
    policy_id = source.split("-")[0] + "-" + source.split("-")[1] if "-" in source else source

    sections: List[Dict[str, Any]] = []
    current_title = None
    current_lines: List[str] = []

    def flush():
        if current_title is not None and current_lines:
            body = "\n".join(current_lines).strip()
            sections.append({
                "policy_id": policy_id,
                "policy_title": doc_title,
                "section_title": current_title,
                "text": f"{doc_title} — {current_title}\n\n{body}",
            })

    for line in lines:
        if line.startswith("## "):
            flush()
            current_title = line[3:].strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)
    flush()
    return sections


def _index_kb_docs() -> None:
    points = []
    policies_dir = KB_DIR / "policies"
    for md_path in sorted(policies_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for section in _split_policy_into_sections(text, md_path.stem):
            point_id = str(uuid.uuid5(_ID_NAMESPACE, f"{md_path.stem}:{section['section_title']}"))
            points.append(PointStruct(
                id=point_id,
                vector=embed_text(section["text"]),
                payload={**section, "source": md_path.name},
            ))
    if points:
        _client.upsert(KB_DOCS_COLLECTION, points=points)
    logger.info("kb_docs indicizzata: %d chunk da %d file di policy", len(points), len(list(policies_dir.glob('*.md'))))


# --------------------------------------------------------------------------
# Indicizzazione — kb_tickets (storico ticket risolti)
# --------------------------------------------------------------------------

def _ticket_to_point(ticket: Dict[str, Any], dataset: str) -> PointStruct:
    # Il testo embeddato è il "lato problema" (subject + description): è
    # quello che una nuova richiesta somiglierà semanticamente, non la
    # risoluzione. La risoluzione va invece nel payload, da mostrare come
    # contesto una volta recuperato il punto.
    embed_source = f"{ticket['subject']}\n\n{ticket['description']}"

    resolution_text = (
        f"Caso simile risolto in precedenza — {ticket['subject']}\n"
        f"Categoria: {ticket['category']} / {ticket['subcategory']} · Priorità: {ticket['priority']}\n"
        f"Risoluzione: {ticket['resolution_summary']}\n"
        f"Passi seguiti: {'; '.join(ticket['resolution_steps'])}\n"
        f"Escalato a un operatore umano: {'sì' if ticket['was_escalated_to_human'] else 'no'}"
        + (f" (motivo: {ticket['escalation_reason']})" if ticket.get("escalation_reason") else "")
    )

    point_id = str(uuid.uuid5(_ID_NAMESPACE, ticket["ticket_id"]))
    return PointStruct(
        id=point_id,
        vector=embed_text(embed_source),
        payload={
            "text": resolution_text,
            "source": ticket["ticket_id"],
            # "real" | "synthetic": permette di filtrare o segmentare in fase di
            # evaluation (es. indicizzare i sintetici e valutare sui reali, o
            # misurare separatamente le due popolazioni).
            "dataset": dataset,
            "category": ticket["category"],
            "subcategory": ticket["subcategory"],
            "priority": ticket["priority"],
            "was_escalated_to_human": ticket["was_escalated_to_human"],
            "escalation_reason": ticket.get("escalation_reason"),
            "tags": ticket.get("tags", []),
        },
    )


def _index_kb_tickets() -> None:
    """
    Indicizza TUTTI i file `past_tickets*.json` presenti in knowledge_base/:
    - `past_tickets.json`           -> ticket reali          (dataset="real")
    - `past_tickets_synthetic.json` -> ticket sintetici      (dataset="synthetic")

    La provenienza finisce nel payload come campo `dataset`, così in fase di
    evaluation si possono separare le due popolazioni senza doverle tenere in
    collection diverse.
    """
    points = []
    for path in sorted(KB_DIR.glob("past_tickets*.json")):
        dataset = "synthetic" if "synthetic" in path.stem else "real"
        tickets = json.loads(path.read_text(encoding="utf-8"))
        points.extend(_ticket_to_point(t, dataset) for t in tickets)
        logger.info("  %s -> %d ticket (dataset=%s)", path.name, len(tickets), dataset)
    if points:
        _client.upsert(KB_TICKETS_COLLECTION, points=points)
    logger.info("kb_tickets indicizzata: %d ticket storici in totale", len(points))


# --------------------------------------------------------------------------
# Setup: crea le collection e indicizza, solo se non è già stato fatto
# --------------------------------------------------------------------------

def ensure_index() -> None:
    if not _client.collection_exists(KB_DOCS_COLLECTION):
        _client.create_collection(
            KB_DOCS_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        _index_kb_docs()

    if not _client.collection_exists(KB_TICKETS_COLLECTION):
        _client.create_collection(
            KB_TICKETS_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        _index_kb_tickets()


# --------------------------------------------------------------------------
# Query — usate dai nodi di retrieval in graph.py
# --------------------------------------------------------------------------

def search_kb_docs(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Ritorna i k chunk di policy più simili a `query`, come lista di dict
    {"text", "source", "score", "policy_id", "section_title", ...}."""
    try:
        results = _client.query_points(KB_DOCS_COLLECTION, query=embed_text(query), limit=k)
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_docs, ritorno lista vuota")
        return []


def search_kb_tickets(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Ritorna i k ticket storici più simili a `query`, stessa forma di search_kb_docs."""
    try:
        results = _client.query_points(KB_TICKETS_COLLECTION, query=embed_text(query), limit=k)
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_tickets, ritorno lista vuota")
        return []


# Costruisce l'indice al primo import del modulo (una volta per processo).
ensure_index()
