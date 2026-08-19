"""
store.py
========
Registro in-memory dei ticket dell'help desk.

È un indice "di lettura" pensato per la UI dell'operatore (coda dei ticket
in attesa di escalation), volutamente SEPARATO dallo stato di esecuzione del
grafo LangGraph (il checkpointer). I due concetti sono diversi:

- il checkpointer di LangGraph sa "a che punto è l'esecuzione del grafo per
  il thread X" — serve per riprendere un interrupt(), non è pensato per
  essere interrogato con query tipo "dammi tutti i ticket aperti";
- il TicketStore rappresenta il record di business "ticket" così come lo
  vedrebbe un vero sistema di help desk: chi l'ha aperto, con che oggetto,
  in che stato. In un sistema reale sarebbe una tabella su database; qui,
  per il prototipo, è un dizionario in RAM (si perde ad ogni riavvio,
  esattamente come InMemorySaver — i due store verranno sostituiti insieme
  quando si introdurrà la persistenza reale).

`ticket_id` coincide sempre con il `thread_id` del grafo LangGraph: sono lo
stesso identificatore usato per due scopi diversi (join implicito tra i due
store, nessuna tabella di mapping necessaria).
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Ticket:
    ticket_id: str                        # == thread_id del grafo LangGraph
    subject: str                          # anteprima: il messaggio che ha innescato l'escalation
    status: str                           # "awaiting_agent" | "resolved"
    review_reason: Optional[str] = None
    confidence: Optional[float] = None
    draft_answer: Optional[str] = None
    resolved_answer: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None


class TicketStore:
    """
    Store in memoria di processo, protetto da un lock: FastAPI (con endpoint
    sincroni `def`, non `async def`) esegue le richieste in un threadpool,
    quindi scritture concorrenti sullo stesso dizionario sono realistiche.
    """

    def __init__(self) -> None:
        self._tickets: Dict[str, Ticket] = {}
        self._lock = threading.Lock()

    def escalate(self, ticket_id: str, subject: str, review_reason: Optional[str],
                 confidence: Optional[float], draft_answer: str) -> None:
        """Registra (o riapre) un ticket come in attesa di un operatore."""
        with self._lock:
            existing = self._tickets.get(ticket_id)
            self._tickets[ticket_id] = Ticket(
                ticket_id=ticket_id,
                subject=subject,
                status="awaiting_agent",
                review_reason=review_reason,
                confidence=confidence,
                draft_answer=draft_answer,
                created_at=existing.created_at if existing else time.time(),
            )

    def remove(self, ticket_id: str) -> None:
        """Rimuove un ticket dal registro senza segnarlo risolto (usato quando
        l'intero thread viene chiuso dall'operatore mentre era ancora in coda)."""
        with self._lock:
            self._tickets.pop(ticket_id, None)

    def resolve(self, ticket_id: str, resolved_answer: str) -> None:
        """Segna un ticket come risolto da un operatore."""
        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if ticket is not None:
                ticket.status = "resolved"
                ticket.resolved_answer = resolved_answer
                ticket.resolved_at = time.time()

    def pending(self) -> List[Ticket]:
        """Ticket attualmente in coda per un operatore, dal più vecchio al più recente."""
        with self._lock:
            items = [t for t in self._tickets.values() if t.status == "awaiting_agent"]
        return sorted(items, key=lambda t: t.created_at)


# Istanza unica condivisa da tutta l'applicazione FastAPI.
ticket_store = TicketStore()
