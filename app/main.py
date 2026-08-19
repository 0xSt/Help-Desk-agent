"""
main.py
=======
App FastAPI che espone il workflow LangGraph (app/graph.py) e serve DUE
interfacce grafiche, entrambe attive in contemporanea sullo stesso processo:

- GET  /        -> interfaccia UTENTE (app/static/index.html): apre un
                    ticket/conversazione, ed eventualmente attende che un
                    operatore lo risolva.
- GET  /agent   -> interfaccia OPERATORE (app/static/agent.html): mostra la
                    coda dei ticket in escalation e permette di editare/
                    approvare la risposta proposta dall'AI.

Le due interfacce condividono lo stesso set di endpoint API:

- POST /api/chat        -> invia un messaggio utente, esegue il grafo.
                            Se il grafo si interrompe per escalation,
                            risponde con status="awaiting_review" e il
                            ticket viene registrato nel TicketStore.
- POST /api/review      -> l'operatore invia la risposta definitiva
                            (modificata o approvata così com'è); il grafo
                            riprende e il ticket viene segnato risolto.
- GET  /api/state/{id}  -> stato corrente di un thread/ticket (usato da
                            entrambe le UI per ricostruirsi dopo un refresh
                            o per fare polling).
- GET  /api/tickets     -> coda dei ticket in attesa di un operatore (usato
                            solo dall'interfaccia operatore).
- GET  /api/threads     -> conversazioni attualmente attive (usato dal
                            selettore lato utente e dalla lista "conversazioni
                            attive" dell'operatore).
- POST /api/threads/{id}/close -> l'operatore chiude definitivamente una
                            conversazione (indipendentemente dal fatto che
                            sia mai passata dalla coda ticket).
"""
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langgraph.types import Command

from app.graph import graph
from app.store import ticket_store
from app.threads import thread_registry
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ReviewRequest,
    TicketQueueResponse,
    TicketSummary,
    ThreadListResponse,
    ThreadSummary,
    CloseThreadResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Help Desk — Demo HITL")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_user_ui():
    """Interfaccia utente: apertura ticket / chat di supporto."""
    return FileResponse("app/static/index.html")


@app.get("/agent")
def serve_agent_ui():
    """Interfaccia operatore: coda ticket in escalation + editor risposta."""
    return FileResponse("app/static/agent.html")


def _pending_interrupt_payload(config: dict):
    """Ritorna il payload dell'interrupt in sospeso per questo thread, o None."""
    snapshot = graph.get_state(config)
    if not snapshot.next:  # tupla vuota = nessun nodo in sospeso = nessun interrupt attivo
        return None
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Un thread chiuso dall'operatore non accetta più nuovi messaggi: il
    # client deve avviarne uno nuovo (thread_id=None). Il controllo va fatto
    # PRIMA di generare un thread_id nuovo, altrimenti un thread_id chiuso
    # esplicitamente passato dal client verrebbe trattato come "nuovo".
    if req.thread_id and thread_registry.is_closed(req.thread_id):
        raise HTTPException(
            status_code=400,
            detail="This conversation has been closed by an agent. Please start a new one.",
        )

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Registra il thread (nuovo o già esistente) nel registro "di lettura"
    # usato dal selettore utente e dalla lista dell'operatore — indipendente
    # dal checkpointer del grafo, vedi app/threads.py.
    thread_registry.touch(thread_id, req.message)

    # Non serve recuperare manualmente la cronologia: il checkpointer di
    # LangGraph la mantiene già associata al thread_id tra una chiamata e
    # l'altra. Basta passare il nuovo messaggio.
    result = graph.invoke({"user_query": req.message}, config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        # Il ticket entra (o rientra) in coda per un operatore: da questo
        # momento è visibile in GET /api/tickets, letto dall'interfaccia
        # operatore. subject = il messaggio che ha innescato l'escalation.
        ticket_store.escalate(
            ticket_id=thread_id,
            subject=req.message,
            review_reason=payload["reason"],
            confidence=payload.get("confidence"),
            draft_answer=payload["draft_answer"],
        )
        return ChatResponse(
            thread_id=thread_id,
            status="awaiting_review",
            draft_answer=payload["draft_answer"],
            review_reason=payload["reason"],
            confidence=payload.get("confidence"),
            history=result.get("history", []),
        )

    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        answer=result["final_answer"],
        history=result.get("history", []),
        reviewed_by_human=result.get("reviewed_by_human", False),
    )


@app.post("/api/review", response_model=ChatResponse)
def review(req: ReviewRequest):
    """Chiamato dall'interfaccia OPERATORE quando invia la risposta definitiva."""
    config = {"configurable": {"thread_id": req.thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=400,
            detail="No review is pending for this thread_id.",
        )

    # Riprende il grafo esattamente dal nodo human_review: il valore passato
    # a Command(resume=...) è ciò che interrupt() restituirà dentro human_review_node.
    result = graph.invoke(Command(resume=req.edited_answer), config)

    # Il ticket esce dalla coda dell'operatore: GET /api/tickets non lo
    # includerà più nella prossima chiamata.
    ticket_store.resolve(req.thread_id, req.edited_answer)

    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        answer=result["final_answer"],
        history=result.get("history", []),
        reviewed_by_human=True,
    )


@app.get("/api/state/{thread_id}", response_model=ChatResponse)
def get_state(thread_id: str):
    """
    Permette a entrambe le interfacce di recuperare lo stato di un thread:
    usato dall'utente per il polling mentre attende un operatore e per
    ricostruire la UI dopo un refresh, e dall'operatore per caricare il
    contesto completo di un ticket selezionato dalla coda.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)

    # snapshot.values è un dizionario vuoto {} se il checkpointer non ha mai
    # visto questo thread_id — lo distinguiamo esplicitamente da un thread
    # concluso, così il frontend può ripulire un thread_id non più valido
    # (es. dopo il riavvio del server, dato che InMemorySaver non persiste).
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Thread not found.")
    payload = _pending_interrupt_payload(config)
    history = snapshot.values.get("history", [])

    if payload is not None:
        return ChatResponse(
            thread_id=thread_id,
            status="awaiting_review",
            draft_answer=payload["draft_answer"],
            review_reason=payload["reason"],
            confidence=payload.get("confidence"),
            history=history,
        )

    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        answer=snapshot.values.get("final_answer"),
        history=history,
        reviewed_by_human=snapshot.values.get("reviewed_by_human", False),
    )


@app.get("/api/tickets", response_model=TicketQueueResponse)
def list_pending_tickets():
    """
    Coda dei ticket in attesa di un operatore, usata SOLO dall'interfaccia
    operatore (app/static/agent.html), interrogata via polling.

    Legge dal TicketStore (non dal checkpointer di LangGraph): vedi
    app/store.py per la motivazione della separazione.
    """
    tickets = ticket_store.pending()
    return TicketQueueResponse(
        tickets=[
            TicketSummary(
                ticket_id=t.ticket_id,
                subject=t.subject,
                review_reason=t.review_reason,
                confidence=t.confidence,
                created_at=t.created_at,
            )
            for t in tickets
        ]
    )


@app.get("/api/threads", response_model=ThreadListResponse)
def list_active_threads():
    """
    Conversazioni attualmente attive (non chiuse). Usato da:
    - interfaccia UTENTE: popola il selettore "le tue conversazioni";
    - interfaccia OPERATORE: popola la lista "conversazioni attive" da cui
      chiudere un thread.
    """
    threads = thread_registry.list_active()
    return ThreadListResponse(
        threads=[
            ThreadSummary(
                thread_id=t.thread_id,
                subject=t.subject,
                last_activity_at=t.last_activity_at,
            )
            for t in threads
        ]
    )


@app.post("/api/threads/{thread_id}/close", response_model=CloseThreadResponse)
def close_thread(thread_id: str):
    """
    Chiude definitivamente un thread — SOLO l'operatore lo chiama.
    Da questo momento /api/chat rifiuterà nuovi messaggi su questo thread_id.

    NOTA/LIMITE: se il thread era ancora fermo su un interrupt() (escalation
    non ancora risolta), lo rimuoviamo dalla coda ticket ma il grafo
    sottostante resta "sospeso per sempre" nel checkpointer — non esiste
    oggi un modo pulito per "annullare" un interrupt pendente in LangGraph.
    Con InMemorySaver non è un problema pratico (si perde comunque al
    riavvio), ma è un punto da rivedere se si introduce un checkpointer
    persistente.
    """
    closed = thread_registry.close(thread_id)
    if not closed:
        raise HTTPException(status_code=404, detail="Thread not found.")

    ticket_store.remove(thread_id)

    return CloseThreadResponse(thread_id=thread_id, status="closed")
