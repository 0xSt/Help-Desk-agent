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

EMBEDDING: di default usa **Gemini** (`gemini-embedding-001`) con i task type
asimmetrici previsti dall'API — `RETRIEVAL_DOCUMENT` per i testi indicizzati,
`RETRIEVAL_QUERY` per le interrogazioni. È una distinzione che incide sulla
qualità: i due tipi producono rappresentazioni pensate per stare ai due lati
della stessa ricerca, e usarne uno solo per entrambi degrada il ranking.

Senza `GEMINI_API_KEY` si ricade su un embedding "hashing trick"
deterministico, puro Python: stesso spirito della modalità mock di llm.py, fa
girare l'intera pipeline senza credenziali ma con qualità semantica modesta
(cattura sovrapposizione lessicale, non significato).

I due provider producono vettori di dimensione diversa (768 contro 256), e
soprattutto **spazi vettoriali incompatibili**: un indice costruito con l'uno
non è interrogabile con l'altro. `ensure_index()` se ne accorge confrontando
la dimensione della collection esistente con quella del provider attivo, e in
caso di mismatch ricostruisce l'indice da zero invece di fallire con un errore
oscuro a query time.
"""
import hashlib
import json
import logging
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app import config
from app.tracing import trace_span

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent / "knowledge_base"
QDRANT_PATH = os.environ.get("QDRANT_PATH", "qdrant_data")

KB_DOCS_COLLECTION = "kb_docs"
KB_TICKETS_COLLECTION = "kb_tickets"

# Task type dell'API di embedding Gemini: asimmetrici per il retrieval.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Namespace fisso per generare ID punto deterministici (uuid5): ri-eseguire
# l'indicizzazione sullo stesso contenuto produce gli stessi ID, quindi
# l'upsert aggiorna i punti esistenti invece di duplicarli.
_ID_NAMESPACE = uuid.UUID("a13a1b2c-6f6e-4c1a-9c1f-0e5b6f3a2b10")

# In locale l'indice sta su disco (nessun server esterno). In Docker Compose
# si valorizza QDRANT_URL e lo stesso client punta al servizio containerizzato:
# cambia solo l'argomento del costruttore.
QDRANT_URL = os.environ.get("QDRANT_URL")
_client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)


# --------------------------------------------------------------------------
# Embedding — provider Gemini, con fallback deterministico offline
# --------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", text.lower())


def _hash_embed(text: str, dim: Optional[int] = None) -> List[float]:
    """
    Embedding deterministico via feature hashing, usato solo senza API key.

    Ogni token finisce in uno di `dim` bucket (indice = hash(token) % dim),
    con segno anch'esso derivato dall'hash; il vettore è normalizzato L2 così
    la similarità coseno si comporta in modo sensato. Cattura sovrapposizione
    lessicale, non significato: serve a far girare la pipeline, non a fare
    retrieval di qualità.

    `dim` viene risolto a ogni chiamata e non come default dell'argomento:
    un default verrebbe legato al valore che la configurazione aveva
    all'import del modulo, ignorando qualunque override successivo.
    """
    dim = dim or config.FALLBACK_EMBEDDING_DIM
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        h = int(digest, 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _l2_normalize(vec: List[float]) -> List[float]:
    """
    Normalizzazione L2 esplicita.

    Serve perché `gemini-embedding-001` restituisce vettori normalizzati alla
    dimensione piena (3072), ma **non** dopo la troncatura Matryoshka a 768:
    i primi 768 valori di un vettore unitario a 3072 dimensioni non formano a
    loro volta un vettore unitario. Senza questo passaggio la similarità
    coseno resterebbe comunque calcolabile, ma i punteggi non sarebbero
    confrontabili tra loro — e le soglie di escalation in config.py si basano
    proprio sul confronto tra punteggi.
    """
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _gemini_embed(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Embedding via API Gemini, a batch.

    `task_type` distingue i due lati della ricerca (RETRIEVAL_DOCUMENT in
    indicizzazione, RETRIEVAL_QUERY in interrogazione): è il parametro che
    dice al modello quale delle due rappresentazioni produrre.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    out: List[List[float]] = []

    for start in range(0, len(texts), config.EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + config.EMBEDDING_BATCH_SIZE]
        response = client.models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=config.EMBEDDING_DIM,
            ),
        )
        out.extend(_l2_normalize(list(e.values)) for e in response.embeddings)

    return out


def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Punto unico di embedding: sceglie il provider e non lascia mai propagare
    un errore di rete al chiamante.

    In caso di fallimento dell'API si ricade sull'embedding di fallback: la
    query produce risultati scadenti ma il turno di conversazione non si
    interrompe. È coerente con la scelta fatta in llm.py — degradare, non
    fallire. Con una differenza importante: se il fallback interviene *in
    indicizzazione* l'indice risulterebbe misto e inutilizzabile, quindi lì
    l'errore viene invece rilanciato (vedi `ensure_index`).
    """
    if not config.GEMINI_API_KEY:
        return [_hash_embed(t) for t in texts]
    return _gemini_embed(texts, task_type)


def embed_query(text: str) -> List[float]:
    """Embedding di una query utente, con degradazione silenziosa in caso di errore."""
    try:
        return embed_texts([text], TASK_QUERY)[0]
    except Exception:
        logger.exception("Embedding della query fallito: uso il fallback deterministico.")
        return _hash_embed(text)


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
    policies_dir = KB_DIR / "policies"
    sections: List[Dict[str, Any]] = []
    for md_path in sorted(policies_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for section in _split_policy_into_sections(text, md_path.stem):
            section["source"] = md_path.name
            section["_key"] = f"{md_path.stem}:{section['section_title']}"
            sections.append(section)

    if not sections:
        return

    # Un'unica chiamata a batch per tutte le sezioni, invece di una per
    # sezione: 60 richieste HTTP diventano 2 (batch da 32).
    vectors = embed_texts([s["text"] for s in sections], TASK_DOCUMENT)

    points = []
    for section, vector in zip(sections, vectors):
        key = section.pop("_key")
        points.append(PointStruct(
            id=str(uuid.uuid5(_ID_NAMESPACE, key)),
            vector=vector,
            payload=section,
        ))
    _client.upsert(KB_DOCS_COLLECTION, points=points)
    logger.info("kb_docs indicizzata: %d chunk da %d file di policy",
                len(points), len(list(policies_dir.glob("*.md"))))


# --------------------------------------------------------------------------
# Indicizzazione — kb_tickets (storico ticket risolti)
# --------------------------------------------------------------------------

def _ticket_embed_text(ticket: Dict[str, Any]) -> str:
    # Il testo embeddato è il "lato problema" (subject + description): è
    # quello a cui una nuova richiesta somiglierà semanticamente, non la
    # risoluzione. La risoluzione va invece nel payload, da mostrare come
    # contesto una volta recuperato il punto.
    return f"{ticket['subject']}\n\n{ticket['description']}"


def _ticket_payload(ticket: Dict[str, Any]) -> Dict[str, Any]:
    resolution_text = (
        f"Similar past ticket — {ticket['subject']}\n"
        f"Category: {ticket['category']} / {ticket['subcategory']} · Priority: {ticket['priority']}\n"
        f"Resolution: {ticket['resolution_summary']}\n"
        f"Steps taken: {'; '.join(ticket['resolution_steps'])}\n"
        f"Escalated to a human agent: {'yes' if ticket['was_escalated_to_human'] else 'no'}"
        + (f" (reason: {ticket['escalation_reason']})" if ticket.get("escalation_reason") else "")
    )
    return {
        "text": resolution_text,
        "source": ticket["ticket_id"],
        "category": ticket["category"],
        "subcategory": ticket["subcategory"],
        "priority": ticket["priority"],
        "was_escalated_to_human": ticket["was_escalated_to_human"],
        "escalation_reason": ticket.get("escalation_reason"),
        "tags": ticket.get("tags", []),
    }


def _index_kb_tickets() -> None:
    """
    Indicizza lo storico ticket (`past_tickets.json`), un punto per ticket.

    L'intero corpus è materiale simulato per questo progetto universitario:
    non c'è distinzione tra ticket "reali" e "sintetici", sono tutti dati di
    scenario trattati allo stesso modo.
    """
    tickets_path = KB_DIR / "past_tickets.json"
    tickets = json.loads(tickets_path.read_text(encoding="utf-8"))
    if not tickets:
        return

    vectors = embed_texts([_ticket_embed_text(t) for t in tickets], TASK_DOCUMENT)
    points = [
        PointStruct(
            id=str(uuid.uuid5(_ID_NAMESPACE, t["ticket_id"])),
            vector=vector,
            payload=_ticket_payload(t),
        )
        for t, vector in zip(tickets, vectors)
    ]
    _client.upsert(KB_TICKETS_COLLECTION, points=points)
    logger.info("kb_tickets indicizzata: %d ticket storici", len(points))


# --------------------------------------------------------------------------
# Setup: crea le collection e indicizza, solo se non è già stato fatto
# --------------------------------------------------------------------------

def _collection_dim(name: str) -> Optional[int]:
    """Dimensione dei vettori di una collection esistente, o None se non esiste."""
    if not _client.collection_exists(name):
        return None
    params = _client.get_collection(name).config.params.vectors
    return getattr(params, "size", None)


def _ensure_collection(name: str, index_fn) -> None:
    """
    Crea e popola la collection se manca, oppure la ricostruisce se la sua
    dimensione non corrisponde al provider di embedding attivo.

    Il controllo sulla dimensione è ciò che rende indolore il passaggio da un
    provider all'altro (o il cambio di `EMBEDDING_DIM`): senza, un indice
    costruito con l'hashing trick a 256 dimensioni verrebbe interrogato con
    vettori Gemini a 768 e Qdrant fallirebbe a query time con un errore
    difficile da ricondurre alla causa. Vettori di provider diversi non sono
    comunque confrontabili, quindi ricostruire è l'unica opzione corretta.
    """
    want = config.active_embedding_dim()
    have = _collection_dim(name)

    if have == want:
        return

    if have is not None:
        logger.warning(
            "Collection '%s' ha vettori a %d dimensioni ma il provider attivo (%s) "
            "ne produce %d: ricostruisco l'indice da zero.",
            name, have, config.embedding_provider(), want,
        )
        _client.delete_collection(name)

    _client.create_collection(
        name,
        vectors_config=VectorParams(size=want, distance=Distance.COSINE),
    )
    try:
        index_fn()
    except Exception:
        # In indicizzazione NON si degrada al fallback: un indice a provider
        # misto sarebbe silenziosamente inutilizzabile. Meglio lasciare la
        # collection vuota e far emergere l'errore.
        logger.exception("Indicizzazione di '%s' fallita: la collection resta vuota.", name)
        raise


def ensure_index() -> None:
    """
    Costruisce l'indice se serve. Chiamata all'import del modulo, e
    richiamabile esplicitamente dal job di ingestion in Docker Compose.
    """
    logger.info("Provider di embedding attivo: %s (%d dimensioni)",
                config.embedding_provider(), config.active_embedding_dim())
    _ensure_collection(KB_DOCS_COLLECTION, _index_kb_docs)
    _ensure_collection(KB_TICKETS_COLLECTION, _index_kb_tickets)


# --------------------------------------------------------------------------
# Query — usate dai nodi di retrieval in graph.py
# --------------------------------------------------------------------------

@trace_span("retrieval.kb_docs")
def search_kb_docs(query: str, k: int = config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
    """Ritorna i k chunk di policy più simili a `query`, come lista di dict
    {"text", "source", "score", "policy_id", "section_title", ...}."""
    try:
        results = _client.query_points(KB_DOCS_COLLECTION, query=embed_query(query), limit=k)
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_docs, ritorno lista vuota")
        return []


@trace_span("retrieval.kb_tickets")
def search_kb_tickets(query: str, k: int = config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
    """Ritorna i k ticket storici più simili a `query`, stessa forma di search_kb_docs."""
    try:
        results = _client.query_points(KB_TICKETS_COLLECTION, query=embed_query(query), limit=k)
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_tickets, ritorno lista vuota")
        return []


# Costruisce l'indice al primo import del modulo (una volta per processo).
ensure_index()
