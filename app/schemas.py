"""
schemas.py
==========
Modelli Pydantic per le richieste/risposte esposte da FastAPI.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: Optional[str] = None  # se assente, ne viene generato uno nuovo
    message: str


class ReviewRequest(BaseModel):
    thread_id: str
    edited_answer: str  # testo finale scelto dall'umano (modificato o approvato tale e quale)


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" oppure "awaiting_review"
    answer: Optional[str] = None         # presente se status == "completed"
    draft_answer: Optional[str] = None   # presente se status == "awaiting_review"
    review_reason: Optional[str] = None
    confidence: Optional[float] = None
    history: Optional[List[Dict[str, str]]] = None  # turni precedenti già conclusi, per ricostruire la UI dopo un refresh
    reviewed_by_human: Optional[bool] = None  # True se 'answer' è stato validato/scritto da un operatore
    # Elenco completo dei trigger che hanno causato l'escalation, ciascuno con
    # la clausola di policy che lo giustifica. L'operatore vede così *tutti* i
    # motivi, non solo il primo riassunto in `review_reason`.
    escalation_triggers: Optional[List[Dict[str, str]]] = None


class TicketSummary(BaseModel):
    """Riga sintetica per la coda dell'operatore (GET /api/tickets)."""
    ticket_id: str
    subject: str
    review_reason: Optional[str] = None
    confidence: Optional[float] = None
    created_at: float


class TicketQueueResponse(BaseModel):
    tickets: List[TicketSummary]


class ThreadSummary(BaseModel):
    """Riga sintetica per il selettore utente e per la lista 'conversazioni attive' dell'operatore."""
    thread_id: str
    subject: str
    last_activity_at: float


class ThreadListResponse(BaseModel):
    threads: List[ThreadSummary]


class CloseThreadResponse(BaseModel):
    thread_id: str
    status: str  # sempre "closed"
