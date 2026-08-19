"""
graph.py
========
Definisce il workflow LangGraph dell'help desk, con:

- due nodi di retrieval che recuperano contesto da due knowledge base
  distinte (policy IT e storico ticket risolti) prima di generare la
  risposta;
- un nodo human-in-the-loop (HITL) per l'escalation a un operatore umano.

Flusso del grafo:

               ┌──────────────────┐
        ┌─────▶│ retrieve_kb_docs  │────┐
        │      └──────────────────┘    │
    START                               ▼
        │      ┌─────────────────────┐ ┌────────┐   needs_review=False   ┌──────────┐
        └─────▶│ retrieve_kb_tickets │▶│ agent  │────────────────────────▶ finalize │──▶ END
               └─────────────────────┘ └────────┘                       └──────────┘
                                            │ needs_review=True                ▲
                                            ▼                                  │
                                     ┌────────────────┐                       │
                                     │ human_review    │───────────────────────┘
                                     │ (interrupt())   │
                                     └────────────────┘

I due nodi di retrieval partono in parallelo da START (fan-out) e convergono
entrambi su `agent` (fan-in): LangGraph esegue `agent` solo dopo che ENTRAMBI
hanno prodotto il loro output nello stesso super-step, perché ciascuno scrive
su una chiave di stato diversa (`kb_docs_context` / `kb_tickets_context`) e
quindi non c'è conflitto di scrittura da risolvere.

Il retrieval vero e proprio (embedding, similarità, query su Qdrant) vive in
app/retrieval.py — qui i due nodi si limitano a invocarlo e a incapsulare
eventuali errori, così un problema di retrieval non fa mai fallire l'intero
turno di conversazione (stesso principio di robustezza già usato in llm.py).
"""
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from app.llm import generate_draft_answer, needs_mandatory_review
from app.retrieval import search_kb_docs, search_kb_tickets

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Stato condiviso che fluisce tra i nodi del grafo per un singolo 'giro'."""
    user_query: str                           # domanda/ticket dell'utente in questo turno
    history: List[Dict[str, str]]             # cronologia dei turni precedenti (persistita dal checkpointer)
    kb_docs_context: List[Dict[str, Any]]     # passaggi di policy recuperati da kb_docs (Qdrant)
    kb_tickets_context: List[Dict[str, Any]]  # ticket storici simili recuperati da kb_tickets (Qdrant)
    draft_answer: str                         # bozza generata dal nodo agent
    confidence: float                         # confidenza auto-dichiarata dal modello (0-1)
    needs_review: bool                        # True -> il grafo passa dal nodo human_review (escalation)
    review_reason: Optional[str]              # motivo mostrato all'operatore
    final_answer: Optional[str]               # risposta definitiva (AI o corretta da operatore)
    reviewed_by_human: bool                   # traccia se questa risposta è passata da un operatore


def retrieve_kb_docs_node(state: AgentState) -> Dict[str, Any]:
    """
    Knowledge base "policy IT" (le 8 policy POL-001..POL-008, spezzate per
    sezione). Recupera i passaggi di policy più rilevanti per la richiesta
    dell'utente — sono il contesto che aiuta l'agente a rispondere in modo
    coerente con le procedure aziendali reali invece di inventare la prassi.
    """
    results = search_kb_docs(state["user_query"], k=3)
    logger.info("retrieve_kb_docs_node -> %d passaggi recuperati", len(results))
    return {"kb_docs_context": results}


def retrieve_kb_tickets_node(state: AgentState) -> Dict[str, Any]:
    """
    Knowledge base "storico ticket risolti". Recupera casi passati simili
    (con relativa risoluzione, categoria ed esito di escalation) da usare
    come precedente concreto nella risposta.
    """
    results = search_kb_tickets(state["user_query"], k=3)
    logger.info("retrieve_kb_tickets_node -> %d ticket simili recuperati", len(results))
    return {"kb_tickets_context": results}


def agent_node(state: AgentState) -> Dict[str, Any]:
    """Nodo 'agente AI': genera la bozza e decide se è necessaria l'escalation."""
    query = state["user_query"]
    result = generate_draft_answer(
        query,
        state.get("history", []),
        kb_docs_context=state.get("kb_docs_context", []),
        kb_tickets_context=state.get("kb_tickets_context", []),
    )

    # NOTA: la logica di escalation è ancora quella "semplice" (regola su
    # parole chiave + soglia di confidenza). Verrà rivista in uno step
    # successivo, probabilmente incorporando anche segnali dal retrieval
    # (es. "nessun passaggio rilevante trovato" -> escalation forzata).
    mandatory = needs_mandatory_review(query)
    low_confidence = result["confidence"] < 0.7
    needs_review = mandatory or low_confidence

    reason = None
    if mandatory:
        reason = "This request involves security or account access and always requires human review."
    elif low_confidence:
        reason = f"The model reported low confidence ({result['confidence']:.2f})."

    logger.info("agent_node -> needs_review=%s (%s)", needs_review, reason)

    return {
        "draft_answer": result["answer"],
        "confidence": result["confidence"],
        "needs_review": needs_review,
        "review_reason": reason,
        "final_answer": None,
    }


def route_after_agent(state: AgentState) -> str:
    """Edge condizionale: instrada verso l'escalation umana o direttamente alla finalizzazione."""
    return "human_review" if state["needs_review"] else "finalize"


def human_review_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo HITL (escalation): `interrupt()` sospende qui l'esecuzione del grafo.

    ATTENZIONE (comportamento di LangGraph da tenere a mente): quando il grafo
    viene ripreso con Command(resume=...), questa funzione viene RI-ESEGUITA
    DALL'INIZIO. Qui non è un problema perché non c'è nulla prima della
    chiamata a interrupt(), ma se in futuro si aggiunge codice con effetti
    collaterali PRIMA di interrupt() in questo nodo, verrebbe eseguito due volte.
    """
    corrected_answer = interrupt({
        "draft_answer": state["draft_answer"],
        "reason": state["review_reason"],
        "confidence": state["confidence"],
    })
    return {"final_answer": corrected_answer, "reviewed_by_human": True}


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Consolida la risposta definitiva e la aggiunge alla cronologia del thread."""
    final = state.get("final_answer")
    reviewed = final is not None
    if not reviewed:
        final = state["draft_answer"]

    history = state.get("history", []) + [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": final},
    ]

    return {
        "final_answer": final,
        "reviewed_by_human": reviewed,
        "history": history,
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("retrieve_kb_docs", retrieve_kb_docs_node)
    builder.add_node("retrieve_kb_tickets", retrieve_kb_tickets_node)
    builder.add_node("agent", agent_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    # Fan-out: le due ricerche partono in parallelo da START...
    builder.add_edge(START, "retrieve_kb_docs")
    builder.add_edge(START, "retrieve_kb_tickets")
    # ...fan-in: 'agent' aspetta che ENTRAMBE abbiano scritto il loro pezzo di stato.
    builder.add_edge("retrieve_kb_docs", "agent")
    builder.add_edge("retrieve_kb_tickets", "agent")

    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"human_review": "human_review", "finalize": "finalize"},
    )
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)

    # InMemorySaver = checkpointer in memoria di processo: ad ogni "super-step"
    # salva lo stato associato al thread_id corrente, cosa che rende possibile
    # sospendere il grafo su interrupt() e riprenderlo più tardi (anche da una
    # richiesta HTTP successiva e indipendente). Si perde tutto se il processo
    # riparte: in produzione andrebbe sostituito con un checkpointer persistente
    # (es. langgraph-checkpoint-postgres o -sqlite).
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Istanza unica del grafo compilato, condivisa da tutta l'applicazione FastAPI.
graph = build_graph()
