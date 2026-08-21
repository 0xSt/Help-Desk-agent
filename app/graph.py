"""
graph.py
========
Definisce il workflow LangGraph dell'help desk:

- due nodi di retrieval che recuperano contesto da due knowledge base
  distinte (policy IT e storico ticket risolti);
- un nodo che genera la bozza e osserva il ticket (`agent`);
- un nodo che decide l'escalation con logica multisegnale (`decide_escalation`);
- un nodo human-in-the-loop per l'escalation vera e propria (`human_review`).

Flusso del grafo:

           ┌──────────────────┐
    ┌─────▶│ retrieve_kb_docs  │────┐
    │      └──────────────────┘    │
  START                             ▼           escalate=False
    │      ┌─────────────────────┐ ┌───────┐   ┌──────────────────┐   ┌──────────┐
    └─────▶│ retrieve_kb_tickets │▶│ agent │──▶│ decide_escalation │──▶│ finalize │──▶ END
           └─────────────────────┘ └───────┘   └──────────────────┘   └──────────┘
                                                      │ escalate=True      ▲
                                                      ▼                    │
                                               ┌────────────────┐          │
                                               │ human_review    │─────────┘
                                               │ (interrupt())   │
                                               └────────────────┘

I due nodi di retrieval partono in parallelo da START (fan-out) e convergono
entrambi su `agent` (fan-in): LangGraph esegue `agent` solo dopo che ENTRAMBI
hanno prodotto il loro output nello stesso super-step, perché ciascuno scrive
su una chiave di stato diversa (`kb_docs_context` / `kb_tickets_context`) e
quindi non c'è conflitto di scrittura da risolvere.

**Perché generazione e decisione sono due nodi distinti.** `agent` produce
bozza, confidenza e segnali; `decide_escalation` applica su quei segnali le
regole di policy. Tenerli separati serve a tre cose concrete: la decisione
diventa misurabile in isolamento nell'evaluation, le sue soglie diventano
iperparametri variabili tra un run e l'altro, e nel tracing MLflow appare
come uno span dedicato con i propri input e output.
"""
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from app import config, escalation
from app.llm import generate_draft_answer
from app.retrieval import search_kb_docs, search_kb_tickets

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Stato condiviso che fluisce tra i nodi del grafo per un singolo 'giro'."""
    user_query: str                           # domanda/ticket dell'utente in questo turno
    history: List[Dict[str, str]]             # cronologia dei turni precedenti (persistita dal checkpointer)
    exclude_sources: List[str]                # id da escludere dal retrieval (leave-one-out in evaluation)
    kb_docs_context: List[Dict[str, Any]]     # passaggi di policy recuperati da kb_docs (Qdrant)
    kb_tickets_context: List[Dict[str, Any]]  # ticket storici simili recuperati da kb_tickets (Qdrant)
    draft_answer: str                         # bozza generata dal nodo agent
    confidence: float                         # confidenza auto-dichiarata dal modello (0-1)
    signals: Dict[str, Any]                   # osservazioni strutturate sul ticket (vedi llm.TicketSignals)
    needs_review: bool                        # True -> il grafo passa dal nodo human_review (escalation)
    review_reason: Optional[str]              # motivo sintetico mostrato all'operatore
    escalation_triggers: List[Dict[str, str]] # elenco completo dei trigger scattati, con clausola di policy
    final_answer: Optional[str]               # risposta definitiva (AI o corretta da operatore)
    reviewed_by_human: bool                   # traccia se questa risposta è passata da un operatore


def retrieve_kb_docs_node(state: AgentState) -> Dict[str, Any]:
    """
    Knowledge base "policy IT" (le 8 policy POL-001..POL-008, spezzate per
    sezione). Recupera i passaggi di policy più rilevanti per la richiesta
    dell'utente — sono il contesto che aiuta l'agente a rispondere in modo
    coerente con le procedure aziendali reali invece di inventare la prassi.
    """
    results = search_kb_docs(
        state["user_query"],
        k=config.RETRIEVAL_TOP_K,
        exclude_sources=state.get("exclude_sources") or None,
    )
    logger.info("retrieve_kb_docs_node -> %d passaggi recuperati", len(results))
    return {"kb_docs_context": results}


def retrieve_kb_tickets_node(state: AgentState) -> Dict[str, Any]:
    """
    Knowledge base "storico ticket risolti". Recupera casi passati simili
    (con relativa risoluzione, categoria ed esito di escalation) da usare
    come precedente concreto nella risposta.
    """
    results = search_kb_tickets(
        state["user_query"],
        k=config.RETRIEVAL_TOP_K,
        # In evaluation si esclude il ticket usato come query: senza, il
        # retrieval lo troverebbe con similarità ~1.0 e il modello leggerebbe
        # nel contesto "Escalated to a human agent: yes", cioè la risposta
        # esatta alla domanda che gli stiamo ponendo. In esercizio normale la
        # chiave è assente e il filtro non viene applicato.
        exclude_sources=state.get("exclude_sources") or None,
    )
    logger.info("retrieve_kb_tickets_node -> %d ticket simili recuperati", len(results))
    return {"kb_tickets_context": results}


def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodo 'agente AI': genera la bozza di risposta e le osservazioni sul ticket.

    NON decide l'escalation: quella spetta a `decide_escalation_node`. La
    separazione è deliberata — il modello interpreta il testo, le regole di
    policy le applica codice deterministico e ispezionabile (vedi il docstring
    di app/escalation.py).
    """
    result = generate_draft_answer(
        state["user_query"],
        state.get("history", []),
        kb_docs_context=state.get("kb_docs_context", []),
        kb_tickets_context=state.get("kb_tickets_context", []),
    )

    logger.info(
        "agent_node -> categoria=%s priorità=%s confidenza=%.2f",
        result.signals.category, result.signals.priority, result.confidence,
    )

    return {
        "draft_answer": result.answer,
        "confidence": result.confidence,
        "signals": result.signals.model_dump(),
        # Reset esplicito: in un turno successivo sullo stesso thread questi
        # campi porterebbero altrimenti i valori del turno precedente.
        "final_answer": None,
        "reviewed_by_human": False,
    }


def decide_escalation_node(state: AgentState) -> Dict[str, Any]:
    """
    Applica la logica multisegnale di escalation (app/escalation.py) ai
    segnali del modello, alla sua confidenza e agli esiti del retrieval.

    È un nodo a sé, e non poche righe dentro `agent_node`, per tre motivi:
    - è la parte che l'evaluation dovrà misurare in isolamento;
    - le sue soglie sono iperparametri da far variare tra un run e l'altro;
    - nel tracing MLflow diventa uno span separato, con i propri input/output,
      così si vede quale segnale ha determinato l'esito di quel ticket.
    """
    from app.llm import TicketSignals

    signals = TicketSignals.model_validate(state["signals"])
    decision = escalation.decide(
        signals=signals,
        confidence=state["confidence"],
        kb_docs=state.get("kb_docs_context", []),
        kb_tickets=state.get("kb_tickets_context", []),
    )

    return {
        "needs_review": decision.escalate,
        "review_reason": decision.reason,
        "escalation_triggers": decision.as_payload(),
    }


def route_after_decision(state: AgentState) -> str:
    """Edge condizionale: instrada verso l'operatore umano o alla finalizzazione."""
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
        # L'elenco completo dei trigger accompagna il ticket fino alla console
        # operatore: POL-006 §5.2 chiede che al ticket escalato sia allegato
        # il contesto completo, incluso il motivo specifico dell'escalation.
        "triggers": state.get("escalation_triggers", []),
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
    builder.add_node("decide_escalation", decide_escalation_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)

    # Fan-out: le due ricerche partono in parallelo da START...
    builder.add_edge(START, "retrieve_kb_docs")
    builder.add_edge(START, "retrieve_kb_tickets")
    # ...fan-in: 'agent' aspetta che ENTRAMBE abbiano scritto il loro pezzo di stato.
    builder.add_edge("retrieve_kb_docs", "agent")
    builder.add_edge("retrieve_kb_tickets", "agent")

    builder.add_edge("agent", "decide_escalation")
    builder.add_conditional_edges(
        "decide_escalation",
        route_after_decision,
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
