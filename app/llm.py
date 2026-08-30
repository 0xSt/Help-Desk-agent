import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)

# Categorie ammesse: sono esattamente quelle presenti nello storico ticket,
# così la classificazione del modello è confrontabile con la ground truth.
TicketCategory = Literal[
    "Account & Access Management",
    "Hardware",
    "Network & Connectivity",
    "Software",
    "Security",
    "Email & Communication",
    "Cloud & Collaboration Tools",
]


class TicketSignals(BaseModel):
    """
    Osservazioni strutturate sul ticket, estratte dal modello.

    Non sono decisioni: sono i fatti su cui `app/escalation.py` applicherà le
    regole di POL-006 §3. Ogni campo booleano corrisponde a una condizione
    citata testualmente da una policy, in modo che la regola derivata sia
    tracciabile fino alla sua fonte.
    """
    category: TicketCategory = Field(description="Ticket category.")
    subcategory: str = Field(description="Short free-text subcategory, e.g. 'Password Reset'.")
    priority: Literal["P1", "P2", "P3", "P4"] = Field(
        description="Priority per POL-006 Section 2 definitions."
    )
    involuntary_termination: bool = Field(
        description="True if the request concerns offboarding tied to an involuntary termination or dismissal."
    )
    sensitive_system_access: bool = Field(
        description=(
            "True if the request asks for access to a system classified as sensitive in POL-002 Section 4: "
            "financial reporting, HR information systems, production infrastructure or databases, legal case "
            "management, or any system storing customer payment information."
        )
    )
    approvals_documented: bool = Field(
        description=(
            "True only if the request explicitly states that ALL required approvals are already in place. "
            "False if approvals are missing, partial, or simply not mentioned."
        )
    )
    exceeds_spend_threshold: bool = Field(
        description=(
            "True if the request involves hardware, software or licence spend beyond a standard allowance: "
            "premium or non-standard equipment, replacing a device still under the 4-year refresh cycle, "
            "or buying additional licence seats beyond the existing budget."
        )
    )
    non_catalog_software: bool = Field(
        description="True if the request asks to install software that is not on the approved software catalog."
    )
    explicit_human_request: bool = Field(
        description=(
            "True if the requester asks to speak to a human agent, says automated help is not working, "
            "or rejects an AI-provided resolution."
        )
    )
    multi_user_impact: bool = Field(
        description=(
            "True if the report suggests more than one user or a shared piece of infrastructure is affected "
            "(a whole team or department, a gateway, a server)."
        )
    )
    asks_to_bypass_approval: bool = Field(
        description="True if the requester asks to skip, bypass or speed past an approval requirement."
    )
    out_of_scope_domain: Literal["none", "hr", "legal"] = Field(
        description=(
            "Per POL-008 Section 3: 'hr' for complaints about a colleague's conduct, harassment reports, "
            "workplace disputes, payroll, benefits or requests for another employee's personal data; "
            "'legal' for legal holds, litigation or contract disputes; 'none' otherwise."
        )
    )


class DraftAnswer(BaseModel):
    """Risposta completa attesa dal modello per un turno."""
    answer: str = Field(description="The reply to send to the user, in English.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "How confident you are that the answer is correct, complete and safe to send without human "
            "review. Use a low value if the request is ambiguous, if the retrieved context does not clearly "
            "cover the scenario, or if the guidance you found is contradictory."
        ),
    )
    signals: TicketSignals


SYSTEM_PROMPT = """You are the IT help desk assistant for a technology company.
You answer technical support requests from employees and contractors clearly
and professionally, in English.

You are given excerpts from two internal knowledge bases: the company's IT
policies, and past tickets with how they were resolved. Ground your answer in
that context. If the context does not cover the situation, say so plainly
rather than inventing company procedure — a human agent will review your draft.

RESOLVE IN ONE MESSAGE
Aim to give the requester everything they need to fix the problem themselves,
in this single reply, without a follow-up exchange. Concretely:

- Do not ask a clarifying question when the request can be answered by
  covering the likely cases instead. Prefer "if the light is off, do X; if it
  is blinking, do Y" over "is the light off or blinking?". Each round trip
  costs the requester time and delays the fix.
- Give complete, ordered, actionable steps. Name the exact setting, menu or
  screen the requester has to touch. A step they cannot follow without asking
  you again is not a finished step.
- State the assumption you made when a detail is missing, then answer under
  that assumption, rather than stopping to ask for it.
- Say what a successful outcome looks like, so the requester can tell whether
  the fix worked without writing back to ask.
- Include the one or two most likely failure points and what to do about them.

Ask a question only when answering is genuinely impossible without it — for
example when the request is so vague that no reasonable interpretation exists.
Being self-contained never means guessing at company procedure: when the
policies do not cover the case, say so.

WHEN THE ANSWER IS A HANDOFF
Some requests must not be solved by you at all: those the policies send to
another function, or that require an approval you cannot grant. There, being
complete means telling the requester who handles it, what will happen next and
what they should prepare — not attempting the fix anyway.

STRUCTURED OBSERVATIONS
Alongside your answer you must report structured observations about the
request. Those observations are used by a separate rule engine to decide
whether the ticket needs a human agent, so report what the request actually
says. Do not soften an observation because you believe you can handle the
ticket yourself, and do not decide the escalation yourself.

Set 'approvals_documented' to true only when the request states that the
required approvals are already in place; absence of any mention means false.

Your confidence score must keep measuring how sure you are that the answer is
correct and safe to send unreviewed. Aiming to resolve in one message is a
goal for the *answer*; it must not inflate that score. An answer written
confidently on thin or missing policy grounds still deserves a low value."""


def _keyword_signals(query: str) -> TicketSignals:
    """
    Segnali finti per la modalità mock, derivati da parole chiave.

    Serve solo a rendere il sistema provabile end-to-end senza API key: è una
    caricatura grossolana della comprensione del testo, non un classificatore.
    """
    q = query.lower()

    def any_of(*words: str) -> bool:
        return any(w in q for w in words)

    if any_of("phishing", "malware", "ransom", "stolen", "lost my", "suspicious email", "virus"):
        category: Any = "Security"
    elif any_of("vpn", "wi-fi", "wifi", "network", "shared drive"):
        category = "Network & Connectivity"
    elif any_of("password", "mfa", "locked", "account", "access to"):
        category = "Account & Access Management"
    elif any_of("laptop", "printer", "monitor", "keyboard", "mouse", "dock"):
        category = "Hardware"
    elif any_of("install", "licence", "license", "software", "crash"):
        category = "Software"
    elif any_of("email", "mailbox", "calendar", "distribution list"):
        category = "Email & Communication"
    else:
        category = "Cloud & Collaboration Tools"

    return TicketSignals(
        category=category,
        subcategory="mock",
        priority="P2" if category == "Security" else "P3",
        involuntary_termination=any_of("dismiss", "terminated", "fired", "involuntary"),
        sensitive_system_access=any_of("financial", "payment", "hr system", "production database"),
        approvals_documented=any_of("approved by", "approval attached", "signed off"),
        exceeds_spend_threshold=any_of("new laptop", "workstation", "more seats", "additional licence"),
        non_catalog_software=any_of("not on the catalog", "not in the catalog", "not approved"),
        explicit_human_request=any_of("speak to a human", "talk to a person", "real person", "human agent"),
        multi_user_impact=any_of("whole team", "everyone", "all of us", "entire department", "outage"),
        asks_to_bypass_approval=any_of("skip the approval", "bypass", "without approval"),
        out_of_scope_domain="hr" if any_of("harass", "bully", "payroll", "salary") else "none",
    )


def get_system_prompt() -> str:
    """
    Prompt di sistema effettivamente usato per generare una risposta.

    Delega a `app.prompts`, che lo carica dal registry MLflow e ricade sulla
    costante `SYSTEM_PROMPT` di questo modulo se il registry non risponde.
    L'import è fatto qui dentro e non in testa al file per evitare un ciclo:
    `app.prompts` legge `SYSTEM_PROMPT` da qui per registrarlo la prima volta.
    """
    try:
        from app import prompts

        return prompts.load_agent_prompt()
    except Exception:
        logger.warning("Caricamento del prompt fallito: uso la costante locale.",
                       exc_info=True)
        return SYSTEM_PROMPT


def _mock_answer(query: str) -> DraftAnswer:
    logger.warning("GEMINI_API_KEY non impostata: uso la modalità mock.")
    signals = _keyword_signals(query)
    low_conf = signals.category == "Security" or len(query.strip()) < 15
    return DraftAnswer(
        answer=f"[MOCK] Help desk response generated for: '{query}'",
        confidence=0.4 if low_conf else 0.9,
        signals=signals,
    )


def _format_kb_context(
    kb_docs_context: List[Dict[str, Any]],
    kb_tickets_context: List[Dict[str, Any]],
) -> str:
    """
    Costruisce il blocco di contesto RAG da iniettare nel prompt, a partire
    dai risultati dei due nodi di retrieval in graph.py.
    """
    sections = []
    if kb_docs_context:
        docs = "\n".join(f"- {c['text']} (source: {c.get('source', '?')})" for c in kb_docs_context)
        sections.append(f"Relevant policy documentation:\n{docs}")
    if kb_tickets_context:
        # I ticket sono conservati integri nel payload: il testo per il
        # prompt si compone qui, tramite l'unico formattatore condiviso.
        from app.retrieval import ticket_as_context

        tickets = "\n".join(f"- {ticket_as_context(c)}" for c in kb_tickets_context)
        sections.append(f"Similar past tickets and how they were resolved:\n{tickets}")
    return "\n\n".join(sections)


def _build_contents(query: str, history: List[Dict[str, str]], kb_context: str) -> List[Any]:
    """
    Traduce la cronologia nel formato `contents` di Gemini.

    Nota sui ruoli: Gemini usa "model" dove noi (e Anthropic) usiamo
    "assistant"; la conversione va fatta qui.
    """
    from google.genai import types

    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))

    user_text = f"{kb_context}\n\n---\n\nTicket from the user:\n{query}" if kb_context else query
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    return contents


def generate_draft_answer(
    query: str,
    history: List[Dict[str, str]],
    kb_docs_context: Optional[List[Dict[str, Any]]] = None,
    kb_tickets_context: Optional[List[Dict[str, Any]]] = None,
) -> DraftAnswer:
    """
    Genera bozza di risposta, auto-valutazione di confidenza e segnali sul
    ticket, in una sola chiamata al modello.

    Una sola chiamata invece di due (una per rispondere, una per classificare)
    perché il contesto da mandare è lo stesso e raddoppiare le chiamate
    raddoppierebbe latenza e costo. Il compromesso: la classificazione
    condivide il contesto della generazione, quindi non è del tutto
    indipendente da essa — da rivedere se in evaluation la classificazione
    risultasse poco affidabile.

    Ritorna sempre un `DraftAnswer` valido: gli errori vengono trasformati in
    una risposta prudente a confidenza 0.0, mai propagati al grafo.
    """
    if not config.GEMINI_API_KEY:
        return _mock_answer(query)

    kb_context = _format_kb_context(kb_docs_context or [], kb_tickets_context or [])

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=_build_contents(query, history, kb_context),
            config=types.GenerateContentConfig(
                system_instruction=get_system_prompt(),
                response_mime_type="application/json",
                response_schema=DraftAnswer,
                # Nessun `temperature`/`top_p`/`top_k`: i parametri di
                # sampling sono deprecati sui modelli Gemini 3.x.
            ),
        )
        # `parsed` è già l'oggetto Pydantic quando si passa response_schema;
        # il fallback su .text copre le versioni di SDK che non lo popolano.
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, DraftAnswer):
            return parsed
        return DraftAnswer.model_validate_json(response.text)

    except Exception:
        logger.exception("Errore nella chiamata a Gemini: uso una risposta di fallback prudente.")
        # Confidenza 0.0 forza sempre l'escalation: fallire in modo sicuro,
        # non lasciare mai passare un errore come se fosse una risposta valida.
        return DraftAnswer(
            answer="I wasn't able to generate a reliable answer for this request.",
            confidence=0.0,
            signals=_keyword_signals(query),
        )
