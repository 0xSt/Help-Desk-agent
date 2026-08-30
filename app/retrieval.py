"""
retrieval.py
============
Indicizzazione e interrogazione delle due knowledge base su **Qdrant**.

Il client si adatta all'ambiente: in sviluppo locale `QdrantClient(path=...)`
tiene l'indice su disco senza bisogno di alcun server, in Docker Compose
`QDRANT_URL` lo fa puntare al servizio containerizzato. Cambia solo
l'argomento del costruttore, il resto del modulo è identico nei due casi.

Due collection:
- `kb_docs`    — le policy IT in app/knowledge_base/policies/*.md, spezzate
                 per sezione (## Titolo).
- `kb_tickets` — i ticket storici in app/knowledge_base/past_tickets.json,
                 un punto per ticket.

INDICIZZAZIONE: `ensure_index()` sincronizza entrambe le collection creandole
se assenti e aggiornandole altrimenti, in modo incrementale (vedi
`sync_collection`). In sviluppo locale viene invocata all'import del modulo
(`AUTO_INDEX`); in Docker Compose è il job dedicato `app/ingest.py` a
chiamarla, prima che il backend accetti richieste.

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
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams,
)

from app import config
from app.tracing import trace_span

logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent / "knowledge_base"

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
_client = (
    QdrantClient(url=config.QDRANT_URL) if config.QDRANT_URL
    else QdrantClient(path=config.QDRANT_PATH)
)


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


def _kb_docs_items() -> List[Dict[str, Any]]:
    """Costruisce i punti da indicizzare per la KB delle policy."""
    items = []
    policies_dir = KB_DIR / "policies"
    for md_path in sorted(policies_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        for section in _split_policy_into_sections(text, md_path.stem):
            section["source"] = md_path.name
            items.append({
                "id": str(uuid.uuid5(_ID_NAMESPACE, f"{md_path.stem}:{section['section_title']}")),
                "text": section["text"],
                "payload": section,
            })
    return items


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


def _kb_tickets_items() -> List[Dict[str, Any]]:
    """Costruisce i punti da indicizzare per la KB dello storico ticket."""
    tickets = json.loads((KB_DIR / "past_tickets.json").read_text(encoding="utf-8"))
    return [
        {
            "id": str(uuid.uuid5(_ID_NAMESPACE, t["ticket_id"])),
            "text": _ticket_embed_text(t),
            "payload": _ticket_payload(t),
        }
        for t in tickets
    ]


# --------------------------------------------------------------------------
# Sincronizzazione incrementale delle collection
# --------------------------------------------------------------------------

# Chiave di payload in cui salviamo l'hash del testo embeddato. È ciò che
# permette di distinguere un punto immutato da uno modificato senza doverlo
# ri-embeddare per scoprirlo.
HASH_FIELD = "_content_hash"


def _content_hash(text: str) -> str:
    """
    Hash del testo embeddato *e* della configurazione di embedding.

    Includere modello e dimensione nell'hash è essenziale: cambiando modello
    lo stesso testo produce un vettore diverso, quindi il punto va rigenerato
    anche se il contenuto non è cambiato di una virgola.
    """
    material = f"{config.embedding_provider()}|{config.GEMINI_EMBEDDING_MODEL}|{config.active_embedding_dim()}|{text}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _collection_dim(name: str) -> Optional[int]:
    """Dimensione dei vettori di una collection esistente, o None se non esiste."""
    if not _client.collection_exists(name):
        return None
    params = _client.get_collection(name).config.params.vectors
    return getattr(params, "size", None)


def _existing_hashes(name: str) -> Dict[str, str]:
    """Mappa {point_id: content_hash} dei punti già presenti in collection."""
    existing: Dict[str, str] = {}
    offset = None
    while True:
        points, offset = _client.scroll(
            name, limit=256, offset=offset,
            with_payload=[HASH_FIELD], with_vectors=False,
        )
        for p in points:
            existing[str(p.id)] = (p.payload or {}).get(HASH_FIELD, "")
        if offset is None:
            break
    return existing


def sync_collection(name: str, items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Porta la collection allo stato descritto da `items`.

    Comportamento richiesto dal deploy: **se la collection non esiste viene
    creata e popolata; se esiste viene aggiornata**, non ricostruita da zero.
    L'aggiornamento è incrementale grazie all'hash del contenuto salvato nel
    payload: si ri-embeddano solo i punti nuovi o modificati, e si cancellano
    quelli spariti dalla sorgente. Su un riavvio senza modifiche alla KB il
    costo in chiamate all'API di embedding è **zero**, il che conta quando
    ogni `docker compose up` ripasserebbe altrimenti da 168 embedding.

    Se la dimensione dei vettori non corrisponde al provider attivo la
    collection viene invece ricreata: vettori di provider diversi vivono in
    spazi incompatibili e non sono mescolabili.
    """
    want_dim = config.active_embedding_dim()
    have_dim = _collection_dim(name)

    if have_dim is not None and have_dim != want_dim:
        logger.warning(
            "Collection '%s': vettori a %d dimensioni ma il provider attivo (%s) "
            "ne produce %d. Ricostruisco da zero.",
            name, have_dim, config.embedding_provider(), want_dim,
        )
        _client.delete_collection(name)
        have_dim = None

    created = have_dim is None
    if created:
        _client.create_collection(
            name, vectors_config=VectorParams(size=want_dim, distance=Distance.COSINE)
        )

    for it in items:
        it["hash"] = _content_hash(it["text"])

    existing = {} if created else _existing_hashes(name)
    to_upsert = [it for it in items if existing.get(it["id"]) != it["hash"]]
    desired_ids = {it["id"] for it in items}
    to_delete = [pid for pid in existing if pid not in desired_ids]

    if to_upsert:
        vectors = embed_texts([it["text"] for it in to_upsert], TASK_DOCUMENT)
        _client.upsert(name, points=[
            PointStruct(
                id=it["id"],
                vector=vec,
                payload={**it["payload"], HASH_FIELD: it["hash"]},
            )
            for it, vec in zip(to_upsert, vectors)
        ])

    if to_delete:
        _client.delete(name, points_selector=to_delete)

    stats = {
        "total": len(items),
        "upserted": len(to_upsert),
        "deleted": len(to_delete),
        "unchanged": len(items) - len(to_upsert),
        "created": int(created),
    }
    logger.info(
        "Collection '%s': %d punti totali (%s, %d aggiornati/nuovi, %d invariati, %d rimossi)",
        name, stats["total"],
        "creata" if created else "aggiornata",
        stats["upserted"], stats["unchanged"], stats["deleted"],
    )
    return stats


def ensure_index() -> Dict[str, Dict[str, int]]:
    """
    Sincronizza entrambe le collection. È il punto d'ingresso usato sia
    dall'auto-indicizzazione locale sia dal job di ingestion in Docker Compose
    (vedi `app/ingest.py`).
    """
    logger.info(
        "Provider di embedding: %s | modello: %s | dimensione: %d",
        config.embedding_provider(), config.GEMINI_EMBEDDING_MODEL, config.active_embedding_dim(),
    )
    return {
        KB_DOCS_COLLECTION: sync_collection(KB_DOCS_COLLECTION, _kb_docs_items()),
        KB_TICKETS_COLLECTION: sync_collection(KB_TICKETS_COLLECTION, _kb_tickets_items()),
    }


# --------------------------------------------------------------------------
# Query — usate dai nodi di retrieval in graph.py
# --------------------------------------------------------------------------


def _exclusion_filter(exclude_sources: Optional[Sequence[str]]):
    """
    Filtro Qdrant che esclude dai risultati i punti con un dato `source`.

    Serve all'evaluation in modalità **leave-one-out**: quando si usa un
    ticket storico come query di test, quel ticket è anche indicizzato, e
    senza esclusione verrebbe recuperato per primo (similarità ~1.0).
    Falserebbe due cose insieme: le metriche di retrieval, ovviamente, ma
    soprattutto la decisione di escalation, perché il payload di un ticket
    contiene l'esito ("Escalated to a human agent: yes") e il modello si
    ritroverebbe nel contesto la risposta esatta alla domanda che gli stiamo
    ponendo. Non misureremmo la sua capacità di decidere, ma di copiare.
    """
    if not exclude_sources:
        return None
    return Filter(must_not=[
        FieldCondition(key="source", match=MatchValue(value=src))
        for src in exclude_sources
    ])


@trace_span("retrieval.kb_docs", span_type="RETRIEVER")
def search_kb_docs(
    query: str,
    k: int = config.RETRIEVAL_TOP_K,
    exclude_sources: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Ritorna i k chunk di policy più simili a `query`, come lista di dict
    {"text", "source", "score", "policy_id", "section_title", ...}."""
    try:
        results = _client.query_points(
            KB_DOCS_COLLECTION,
            query=embed_query(query),
            limit=k,
            query_filter=_exclusion_filter(exclude_sources),
        )
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_docs, ritorno lista vuota")
        return []


@trace_span("retrieval.kb_tickets", span_type="RETRIEVER")
def search_kb_tickets(
    query: str,
    k: int = config.RETRIEVAL_TOP_K,
    exclude_sources: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Ritorna i k ticket storici più simili a `query`, stessa forma di search_kb_docs."""
    try:
        results = _client.query_points(
            KB_TICKETS_COLLECTION,
            query=embed_query(query),
            limit=k,
            query_filter=_exclusion_filter(exclude_sources),
        )
        return [{**p.payload, "score": p.score} for p in results.points]
    except Exception:
        logger.exception("Retrieval fallito su kb_tickets, ritorno lista vuota")
        return []


# Auto-indicizzazione all'import: comoda in sviluppo locale (avvii uvicorn e
# l'indice c'è), ma va disattivata in Docker Compose, dove è un servizio di
# ingestion dedicato a popolare Qdrant prima che il backend parta.
if config.AUTO_INDEX:
    ensure_index()
