"""
escalation.py
=============
Decisione di escalation a un operatore umano, a partire da più segnali.

PRINCIPIO DI PROGETTO: **il modello osserva, il codice decide.**
`llm.py` estrae segnali strutturati dal ticket (categoria, presenza di
approvazioni, impatto multi-utente, ...); qui applichiamo regole
deterministiche su quei segnali. Il motivo non è sfiducia nel modello: è che
POL-006 §3 impone certe escalation *"regardless of the AI agent's confidence
level"*. Una regola del genere non è implementabile se è il modello stesso a
decidere — dev'essere codice ispezionabile, che nessuna risposta del modello
può aggirare.

TRE FAMIGLIE DI SEGNALI, in ordine di forza:

1. **Mandatori (POL-006 §3 e POL-008 §5)** — condizioni sul contenuto del
   ticket. Se una scatta, si escala punto e basta: nessun altro segnale può
   annullarla, per esplicita disposizione della policy.

2. **Confidenza (POL-006 §4, primo criterio)** — il modello dichiara una
   confidenza sotto soglia. Soglia 0.65, presa alla lettera dalla policy.

3. **Retrieval (POL-006 §4, secondo criterio)** — due segnali distinti:
   - *grounding*: nessun passaggio di policy né ticket storico sopra la
     similarità minima, cioè la risposta non poggia su nulla di documentato;
   - *precedente*: tra i ticket storici davvero simili, la maggioranza fu
     escalata da un operatore umano. È il segnale che sfrutta il fatto che lo
     storico contiene la decisione presa in casi analoghi.

Ogni trigger porta con sé il riferimento alla clausola che lo giustifica:
serve all'operatore (che vede *perché* gli è arrivato il ticket) e
all'evaluation (che può misurare l'accuratezza per singolo trigger, non solo
quella complessiva).
"""
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app import config
from app.llm import TicketSignals
from app.tracing import trace_span

logger = logging.getLogger(__name__)

# Categorie che POL-005 §8 dichiara sempre da escalare, senza eccezioni e
# senza possibilità di override da parte della confidenza del modello.
ALWAYS_ESCALATE_CATEGORIES = {"Security"}


@dataclass
class Trigger:
    """Una singola ragione per cui il ticket va escalato."""
    code: str          # riferimento alla clausola, es. "POL-006 §3.1"
    kind: str          # "mandatory" | "confidence" | "retrieval"
    description: str   # testo mostrato all'operatore


@dataclass
class EscalationDecision:
    escalate: bool
    triggers: List[Trigger] = field(default_factory=list)

    @property
    def reason(self) -> Optional[str]:
        """Motivo sintetico da mostrare all'operatore e da salvare sul ticket."""
        if not self.triggers:
            return None
        head = self.triggers[0]
        extra = f" (+{len(self.triggers) - 1} altri segnali)" if len(self.triggers) > 1 else ""
        return f"{head.description} [{head.code}]{extra}"

    def as_payload(self) -> List[Dict[str, str]]:
        """Forma serializzabile, per lo stato del grafo e le risposte API."""
        return [asdict(t) for t in self.triggers]


# --------------------------------------------------------------------------
# 1. Trigger mandatori — POL-006 §3, POL-008 §5
# --------------------------------------------------------------------------

def _mandatory_triggers(signals: TicketSignals) -> List[Trigger]:
    triggers: List[Trigger] = []

    if signals.category in ALWAYS_ESCALATE_CATEGORIES:
        triggers.append(Trigger(
            "POL-006 §3.1", "mandatory",
            "Security-classified ticket: always reviewed by a human agent.",
        ))

    if signals.involuntary_termination:
        triggers.append(Trigger(
            "POL-006 §3.2", "mandatory",
            "Offboarding tied to an involuntary termination: requires immediate action and audit logging.",
        ))

    # Accesso a sistema sensibile: escala se le approvazioni non risultano
    # tutte documentate. POL-002 §4 permette all'agente di *preparare* la
    # richiesta, mai di concedere l'accesso di propria iniziativa.
    if signals.sensitive_system_access and not signals.approvals_documented:
        triggers.append(Trigger(
            "POL-006 §3.3", "mandatory",
            "Access to a sensitive system without documented dual approval.",
        ))

    if signals.exceeds_spend_threshold and not signals.approvals_documented:
        triggers.append(Trigger(
            "POL-006 §3.4", "mandatory",
            "Hardware, software or licence request above the standard threshold without documented approval.",
        ))

    if signals.non_catalog_software:
        triggers.append(Trigger(
            "POL-004 §3", "mandatory",
            "Non-catalog software must be reviewed by the Software Approval Board before installation.",
        ))

    if signals.explicit_human_request:
        triggers.append(Trigger(
            "POL-006 §3.5", "mandatory",
            "The requester explicitly asked for a human agent or rejected automated handling.",
        ))

    if signals.multi_user_impact:
        triggers.append(Trigger(
            "POL-006 §3.7", "mandatory",
            "Possible multi-user or infrastructure-wide impact.",
        ))

    if signals.out_of_scope_domain != "none":
        where = "HR" if signals.out_of_scope_domain == "hr" else "Legal"
        triggers.append(Trigger(
            "POL-008 §5", "mandatory",
            f"Request falls outside standard IT ticket handling and must be redirected to {where}.",
        ))

    if signals.asks_to_bypass_approval:
        triggers.append(Trigger(
            "POL-008 §5", "mandatory",
            "The request asks to bypass an approval requirement defined in another policy.",
        ))

    return triggers


# --------------------------------------------------------------------------
# 2. Confidenza — POL-006 §4, primo criterio
# --------------------------------------------------------------------------

def _confidence_trigger(confidence: float) -> Optional[Trigger]:
    if confidence >= config.CONFIDENCE_THRESHOLD:
        return None
    return Trigger(
        "POL-006 §4", "confidence",
        f"Model confidence {confidence:.2f} is below the {config.CONFIDENCE_THRESHOLD:.2f} threshold.",
    )


# --------------------------------------------------------------------------
# 3. Retrieval — POL-006 §4, secondo criterio
# --------------------------------------------------------------------------

def _best_score(results: List[Dict[str, Any]]) -> float:
    return max((r.get("score", 0.0) for r in results), default=0.0)


def _retrieval_triggers(
    kb_docs: List[Dict[str, Any]],
    kb_tickets: List[Dict[str, Any]],
) -> List[Trigger]:
    triggers: List[Trigger] = []

    best_doc, best_ticket = _best_score(kb_docs), _best_score(kb_tickets)

    # Grounding: la policy parla di "no result above the minimum similarity
    # threshold" da ENTRAMBE le knowledge base — quindi il trigger scatta solo
    # se nessuna delle due offre un appiglio. Se almeno una regge, l'agente ha
    # una base documentale su cui rispondere.
    if best_doc < config.MIN_RETRIEVAL_SCORE and best_ticket < config.MIN_RETRIEVAL_SCORE:
        triggers.append(Trigger(
            "POL-006 §4", "retrieval",
            f"No policy passage or past ticket retrieved above the minimum similarity "
            f"({config.MIN_RETRIEVAL_SCORE:.2f}); best scores were "
            f"{best_doc:.2f} and {best_ticket:.2f}. No documented basis for an automated answer.",
        ))

    # Precedente: tra i ticket storici *davvero* simili, com'è andata?
    similar = [t for t in kb_tickets if t.get("score", 0.0) >= config.PRECEDENT_SCORE_FLOOR]
    if similar:
        escalated = sum(1 for t in similar if t.get("was_escalated_to_human"))
        ratio = escalated / len(similar)
        if ratio >= config.PRECEDENT_ESCALATION_RATIO:
            triggers.append(Trigger(
                "POL-006 §6", "retrieval",
                f"{escalated} of {len(similar)} closely matching past tickets were escalated by a human agent.",
            ))

    return triggers


# --------------------------------------------------------------------------
# Decisione
# --------------------------------------------------------------------------

@trace_span("escalation.decide")
def decide(
    signals: TicketSignals,
    confidence: float,
    kb_docs: List[Dict[str, Any]],
    kb_tickets: List[Dict[str, Any]],
) -> EscalationDecision:
    """
    Combina i segnali in una decisione.

    Regola di combinazione: **OR su tutti i trigger**. È deliberatamente
    conservativa e riflette l'asimmetria dei costi — non escalare un ticket
    che andava escalato (mancata revisione di un incidente di sicurezza, un
    accesso concesso senza approvazione) è molto più grave dell'escalare un
    ticket che l'agente avrebbe saputo chiudere, il cui unico costo è il tempo
    di un operatore.

    I trigger sono ordinati per forza decrescente (mandatori, poi confidenza,
    poi retrieval) così `reason` cita sempre il motivo più cogente.
    """
    triggers = _mandatory_triggers(signals)

    conf = _confidence_trigger(confidence)
    if conf:
        triggers.append(conf)

    triggers.extend(_retrieval_triggers(kb_docs, kb_tickets))

    decision = EscalationDecision(escalate=bool(triggers), triggers=triggers)
    logger.info(
        "escalation: %s (%d trigger: %s)",
        "SÌ" if decision.escalate else "no",
        len(triggers),
        ", ".join(t.code for t in triggers) or "-",
    )
    return decision
