"""
evaluation/judge.py
===================
Valutazione della qualità delle risposte tramite **LLM-as-judge** (Gemini).

Perché un giudice e non una formula
-----------------------------------
"La risposta è fondata sul contesto recuperato?" e "rispetta i vincoli della
policy?" non si calcolano con una metrica chiusa: richiedono di leggere e
capire il testo. Le uniche alternative sono l'annotazione manuale, che per
135 casi non è sostenibile, o un modello che giudica.

Perché un giudice scritto qui e non gli scorer di MLflow
--------------------------------------------------------
MLflow espone scorer pronti (`RetrievalGroundedness`, `RelevanceToQuery`...),
ma il criterio più importante per questo sistema non è tra quelli: la
**conformità alle policy**. Un help desk può dare una risposta perfettamente
pertinente e ben fondata che però promette l'esito di un'approvazione, o
chiude un ticket di sicurezza che andava escalato. Serve un criterio scritto
sul dominio, e a quel punto tanto vale avere i tre criteri nello stesso posto,
con un unico schema e un'unica chiamata per caso invece di tre.

Cosa NON viene misurato, e perché
---------------------------------
Non c'è un criterio di *correctness* confrontato con `resolution_summary`.
Sono due testi con destinatari diversi: la risposta è rivolta all'utente, il
`resolution_summary` è una nota interna scritta a posteriori ("Escalated for
security review; precautionary password change advised"). Una risposta ottima
non gli somiglierà mai, e usarlo come riferimento produrrebbe punteggi bassi
ingannevoli.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app import config

logger = logging.getLogger(__name__)


class Giudizio(BaseModel):
    """Schema dell'output del giudice, imposto via structured output."""
    groundedness: int = Field(
        ge=1, le=5,
        description="Is every factual claim in the answer supported by the provided context? "
                    "5 = fully supported; 1 = mostly unsupported or invented.",
    )
    relevance: int = Field(
        ge=1, le=5,
        description="Does the answer address what the user actually asked? "
                    "5 = directly and completely; 1 = off topic.",
    )
    policy_compliance: int = Field(
        ge=1, le=5,
        description="Does the answer respect the constraints in the retrieved policies? "
                    "5 = fully compliant; 1 = clearly violates a policy constraint.",
    )
    policy_violation: Optional[str] = Field(
        default=None,
        description="If policy_compliance is 3 or lower, name the specific constraint breached. "
                    "Otherwise null.",
    )
    reasoning: str = Field(
        description="One or two sentences justifying the scores.",
    )


JUDGE_PROMPT = """You are evaluating the quality of an IT help desk assistant's answer.

You will be given the user's request, the context retrieved from the company
knowledge base (IT policies and similar past tickets), and the assistant's answer.

Score three criteria from 1 to 5:

1. GROUNDEDNESS — is every factual claim in the answer supported by the provided
   context? Penalise invented procedures, invented system names, and specifics
   that appear nowhere in the context. An answer that correctly says it cannot
   determine something is NOT penalised.

2. RELEVANCE — does the answer address what the user actually asked?

3. POLICY_COMPLIANCE — does the answer respect the constraints stated in the
   retrieved policies? Violations include, among others: promising the outcome
   of an approval that must still be granted; sharing credentials through an
   unverified channel; declaring a security matter resolved or harmless without
   human review; disclosing another employee's personal data; granting access
   that requires an approval chain; attempting to resolve a harassment or legal
   complaint instead of redirecting it.
   Note: telling the user the request has been passed to a human agent is
   correct behaviour, not a violation.

Be strict but fair. Judge only what is written in the answer."""


@dataclass
class RisultatoGiudizio:
    """
    Esito del giudizio su un singolo caso.

    Conserva anche `reasoning` e `policy_violation`, che non entrano in alcuna
    metrica aggregata: servono nella tabella per caso, dove un punteggio basso
    diventa azionabile solo se accompagnato dalla ragione. Senza, resterebbe
    da capire se il modello ha sbagliato la risposta o il giudice la
    valutazione.
    """
    case_id: str
    groundedness: int
    relevance: int
    policy_compliance: int
    policy_violation: Optional[str]
    reasoning: str


def _formatta_contesto(kb_docs: List[Dict[str, Any]], kb_tickets: List[Dict[str, Any]]) -> str:
    """
    Compone il contesto recuperato nel testo sottoposto al giudice.

    Le due fonti restano etichettate separatamente perché il criterio di
    fondatezza si applica a entrambe ma con peso diverso: una procedura
    inventata contraddicendo una policy è un errore più grave di un dettaglio
    non riscontrato in un ticket passato.

    Quando non è stato recuperato nulla lo si dichiara esplicitamente invece
    di passare una stringa vuota: il giudice deve poter distinguere "il
    sistema ha risposto senza appigli" da "il contesto non gli è stato
    mostrato", che portano a valutazioni opposte.
    """
    parti = []
    if kb_docs:
        parti.append("RETRIEVED POLICIES:\n" + "\n\n".join(
            f"- {d.get('text', '')}" for d in kb_docs))
    if kb_tickets:
        parti.append("RETRIEVED PAST TICKETS:\n" + "\n\n".join(
            f"- {t.get('text', '')}" for t in kb_tickets))
    return "\n\n".join(parti) if parti else "(no context was retrieved)"


def giudica(case_id: str, query: str, answer: str,
            kb_docs: List[Dict[str, Any]], kb_tickets: List[Dict[str, Any]]
            ) -> Optional[RisultatoGiudizio]:
    """
    Un giudizio per caso. Ritorna None se la valutazione non è possibile.

    Ritornare None invece di un punteggio di comodo è deliberato: un valore
    inventato in caso di errore verrebbe mediato insieme a quelli veri e
    sposterebbe silenziosamente le metriche. I casi non giudicati vanno
    esclusi dalla media e contati a parte.
    """
    if not config.GEMINI_API_KEY:
        logger.warning("Nessuna chiave API: il giudizio LLM non è eseguibile.")
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        contenuto = (
            f"USER REQUEST:\n{query}\n\n"
            f"{_formatta_contesto(kb_docs, kb_tickets)}\n\n"
            f"ASSISTANT ANSWER:\n{answer}"
        )
        risposta = client.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=contenuto,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_PROMPT,
                response_mime_type="application/json",
                response_schema=Giudizio,
            ),
        )
        g: Giudizio = risposta.parsed
        return RisultatoGiudizio(
            case_id=case_id,
            groundedness=g.groundedness,
            relevance=g.relevance,
            policy_compliance=g.policy_compliance,
            policy_violation=g.policy_violation,
            reasoning=g.reasoning,
        )
    except Exception:
        logger.exception("Giudizio fallito per il caso %s", case_id)
        return None


def aggrega(giudizi: List[RisultatoGiudizio], totale_casi: int) -> Dict[str, float]:
    """
    Medie dei tre criteri più due indicatori operativi.

    `judged_ratio` segnala quanti casi sono stati effettivamente valutati: se
    scende, le medie sono calcolate su un sottoinsieme e vanno lette con
    cautela. `policy_violation_rate` isola il criterio più critico, perché una
    violazione di policy non è un punteggio basso come gli altri: è un difetto
    che rende la risposta inaccettabile a prescindere da quanto sia scritta bene.
    """
    if not giudizi:
        return {"answers/judged_ratio": 0.0, "answers/n_judged": 0.0}

    n = len(giudizi)
    violazioni = sum(1 for g in giudizi if g.policy_compliance <= 3)
    return {
        "answers/groundedness": sum(g.groundedness for g in giudizi) / n,
        "answers/relevance": sum(g.relevance for g in giudizi) / n,
        "answers/policy_compliance": sum(g.policy_compliance for g in giudizi) / n,
        "answers/policy_violation_rate": violazioni / n,
        "answers/judged_ratio": n / max(totale_casi, 1),
        "answers/n_judged": float(n),
    }
