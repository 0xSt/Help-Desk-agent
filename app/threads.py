"""
threads.py
==========
Registro di TUTTE le conversazioni (thread) aperte dagli utenti, a
prescindere dal fatto che siano mai state escalate a un operatore.

Serve due scopi distinti:
- l'interfaccia UTENTE lo usa (GET /api/threads) per proporre un selettore
  di conversazioni attive tra cui scegliere, invece di dover ricordare o
  salvare manualmente un thread_id;
- l'interfaccia OPERATORE lo usa per chiudere definitivamente una
  conversazione (POST /api/threads/{id}/close) — azione indipendente
  dall'aver risolto o meno un'eventuale escalation: un operatore può voler
  chiudere una conversazione anche mai passata dalla coda ticket.

Come TicketStore (vedi store.py), è un registro "di lettura/business"
separato dal checkpointer di LangGraph, per lo stesso motivo: il
checkpointer non è pensato per essere interrogato con query come "elenca
tutte le conversazioni attive". `thread_id` coincide sempre con il
thread_id usato per invocare il grafo.

LIMITE NOTO: non c'è alcuna nozione di identità utente (nessuna
autenticazione nel prototipo) — l'elenco è quindi globale, visibile a
chiunque apra l'interfaccia. Va tenuto presente quando si introdurrà
un'autenticazione reale.
"""
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ThreadInfo:
    thread_id: str
    subject: str                  # anteprima: il primo messaggio inviato su questo thread
    status: str                   # "active" | "closed"
    created_at: float
    last_activity_at: float
    closed_at: Optional[float] = None


class ThreadRegistry:
    """Store in memoria di processo, protetto da un lock (vedi store.py per il perché)."""

    def __init__(self) -> None:
        self._threads: Dict[str, ThreadInfo] = {}
        self._lock = threading.Lock()

    def touch(self, thread_id: str, message: str) -> None:
        """Registra un nuovo thread al primo messaggio, o aggiorna l'attività di uno esistente."""
        with self._lock:
            existing = self._threads.get(thread_id)
            now = time.time()
            if existing is None:
                self._threads[thread_id] = ThreadInfo(
                    thread_id=thread_id,
                    subject=message,
                    status="active",
                    created_at=now,
                    last_activity_at=now,
                )
            else:
                existing.last_activity_at = now

    def is_closed(self, thread_id: str) -> bool:
        with self._lock:
            info = self._threads.get(thread_id)
            return info is not None and info.status == "closed"

    def close(self, thread_id: str) -> bool:
        """Chiude un thread. Ritorna False se il thread_id non è mai stato visto."""
        with self._lock:
            info = self._threads.get(thread_id)
            if info is None:
                return False
            info.status = "closed"
            info.closed_at = time.time()
            return True

    def list_active(self) -> List[ThreadInfo]:
        """Thread attivi, dal più recentemente utilizzato al meno recente."""
        with self._lock:
            items = [t for t in self._threads.values() if t.status == "active"]
        return sorted(items, key=lambda t: t.last_activity_at, reverse=True)


# Istanza unica condivisa da tutta l'applicazione FastAPI.
thread_registry = ThreadRegistry()
