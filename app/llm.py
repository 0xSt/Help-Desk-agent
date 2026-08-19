"""
llm.py
======
Wrapper attorno al modello linguistico (Claude via API Anthropic), con la
persona dell'assistente di help desk IT (azienda informatica fittizia, senza
nome). Se la variabile d'ambiente ANTHROPIC_API_KEY non è impostata, il modulo entra
automaticamente in "modalità mock": produce risposte finte ma coerenti, utile
per provare l'intero flusso (incluso l'interrupt/resume di LangGraph) senza
possedere una chiave API.
"""
import os
import json
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

MODEL_NAME = "claude-sonnet-4-6"

# Argomenti che richiedono SEMPRE un'escalation a un operatore umano,
# indipendentemente dalla confidenza dichiarata dal modello. È un controllo
# deterministico che integra l'auto-valutazione dell'LLM: in un help desk IT
# reale non ci si fida solo del giudizio del modello su se stesso per
# richieste che toccano sicurezza, accessi o dati sensibili.
#
# NOTA: questa è ancora la logica di escalation "semplice" del prototipo
# iniziale. Sarà riprogettata in uno step successivo (vedi graph.py).
# Le parole chiave sono in inglese: knowledge base e utenti parlano inglese.
MANDATORY_REVIEW_KEYWORDS = [
    "admin password", "vpn", "delete account",
    "sensitive data", "breach", "root access", "admin credentials",
    "software license", "unauthorized access",
]

SYSTEM_PROMPT = """You are the IT help desk assistant for a technology company.
Respond to technical support requests from users (employees or contractors)
clearly and professionally, in English.

ALWAYS respond with ONLY a valid JSON object, no extra text, markdown, or
backticks, with exactly this structure:
{"answer": "<your answer in English>", "confidence": <number between 0 and 1>}

"confidence" should reflect how sure you are that the answer is correct,
complete, and safe to send directly to the user without human review. Use a
low value (below 0.7) if the request is ambiguous, involves a specific
configuration you don't know with certainty, or touches on
security/access/sensitive data."""


def needs_mandatory_review(query: str) -> bool:
    """Controllo deterministico e trasparente: alcuni argomenti vanno sempre in escalation."""
    q = query.lower()
    return any(kw in q for kw in MANDATORY_REVIEW_KEYWORDS)


def _mock_answer(query: str) -> Dict:
    """Risposta finta usata quando manca ANTHROPIC_API_KEY, per demo/test offline."""
    logger.warning("ANTHROPIC_API_KEY non impostata: uso la modalità mock.")
    low_conf = needs_mandatory_review(query) or len(query.strip()) < 15
    return {
        "answer": f"[MOCK] Help desk response generated for: '{query}'",
        "confidence": 0.4 if low_conf else 0.9,
    }


def _format_kb_context(kb_docs_context: List[Dict[str, Any]], kb_tickets_context: List[Dict[str, Any]]) -> str:
    """
    Costruisce il blocco di contesto RAG da iniettare nel prompt, a partire
    dai risultati reali dei due nodi di retrieval in graph.py (vedi
    app/retrieval.py). Ogni elemento atteso è un dict con almeno le chiavi
    {"text": ..., "source": ...}. Le etichette delle sezioni sono in inglese
    perché finiscono nel prompt mandato al modello, che risponde in inglese.
    """
    sections = []
    if kb_docs_context:
        docs = "\n".join(f"- {c['text']} (source: {c.get('source', '?')})" for c in kb_docs_context)
        sections.append(f"Relevant policy documentation:\n{docs}")
    if kb_tickets_context:
        tickets = "\n".join(f"- {c['text']} (source: {c.get('source', '?')})" for c in kb_tickets_context)
        sections.append(f"Similar past tickets and how they were resolved:\n{tickets}")
    return "\n\n".join(sections)


def generate_draft_answer(
    query: str,
    history: List[Dict[str, str]],
    kb_docs_context: Optional[List[Dict[str, Any]]] = None,
    kb_tickets_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict:
    """
    Genera una bozza di risposta più un'auto-valutazione di confidenza (0-1).

    `kb_docs_context` e `kb_tickets_context` sono il contesto recuperato dai
    due nodi di retrieval in graph.py (vedi app/retrieval.py) — passaggi di
    policy e ticket storici simili, già pronti per essere iniettati nel prompt.

    Ritorna sempre un dict {"answer": str, "confidence": float}, così che
    graph.py non debba mai gestire eccezioni provenienti da qui.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _mock_answer(query)

    kb_context = _format_kb_context(kb_docs_context or [], kb_tickets_context or [])
    user_content = f"{kb_context}\n\n---\n\n{query}" if kb_context else query

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_content})

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw_text = response.content[0].text
        parsed = json.loads(raw_text)
        return {
            "answer": str(parsed["answer"]),
            "confidence": float(parsed["confidence"]),
        }
    except Exception:
        logger.exception("Errore nella chiamata a Claude: uso una risposta di fallback prudente.")
        # Confidenza 0.0 forza sempre l'escalation: fallire in modo sicuro,
        # non lasciare mai passare un errore silenziosamente come se fosse una risposta valida.
        return {
            "answer": "I wasn't able to generate a reliable answer for this request.",
            "confidence": 0.0,
        }
