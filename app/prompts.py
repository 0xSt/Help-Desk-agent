"""
prompts.py
==========
Versionamento e monitoraggio dei prompt tramite il **Prompt Registry** di
MLflow.

Il problema che risolve
-----------------------
Il prompt è un parametro del sistema al pari di una soglia, ma è l'unico che
non compare da nessuna parte quando si confrontano due esecuzioni. Se la
qualità delle risposte cambia fra una settimana e l'altra, senza versionare il
prompt non si distingue una variazione dovuta alla riformulazione delle
istruzioni da una dovuta al modello, ai dati o alle soglie.

È un rischio concreto in questo progetto: la modifica "risolvi in un solo
messaggio" tocca il testo generato e potenzialmente la confidenza dichiarata,
cioè proprio i segnali su cui si regge l'escalation basata su POL-006 §4.
Senza versionamento, l'effetto di quella modifica non sarebbe attribuibile.

Come funziona
-------------
Ogni prompt è registrato con un nome stabile. `ensure_registered` confronta il
testo corrente con l'ultima versione registrata: **crea una nuova versione solo
se il testo è effettivamente cambiato**. Senza questo confronto ogni riavvio
del servizio produrrebbe una versione nuova identica alla precedente, e la
cronologia diventerebbe illeggibile proprio quando serve.

La versione risultante viene poi registrata come parametro dei run — di avvio
del servizio e di valutazione — così ogni insieme di metriche resta
attribuibile al prompt che l'ha prodotto.

L'alias `production` punta sempre alla versione attiva: permette di caricare il
prompt corrente senza conoscerne il numero.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Nomi stabili nel registry. Cambiarli spezza la cronologia delle versioni.
AGENT_PROMPT_NAME = "helpdesk-agent-system"
JUDGE_PROMPT_NAME = "helpdesk-judge-system"

ALIAS = "production"

# Cache di processo: evita di interrogare il registry a ogni richiesta.
_cache: Dict[str, Optional[Any]] = {}


def ensure_registered(name: str, template: str,
                      commit_message: str = "aggiornamento automatico") -> Optional[Any]:
    """
    Garantisce che `template` sia registrato come versione corrente di `name`.

    Ritorna la versione registrata, oppure `None` se il registry non è
    raggiungibile. Il valore `None` non è un errore da propagare: il prompt
    monitoring è telemetria, e la sua indisponibilità non deve impedire al
    sistema di rispondere ai ticket.

    Il confronto con la versione esistente è ciò che rende l'operazione
    idempotente: registrare a ogni avvio un testo immutato creerebbe versioni
    duplicate che rendono inutile la cronologia.
    """
    if name in _cache:
        return _cache[name]

    try:
        import mlflow.genai as genai

        corrente = None
        try:
            corrente = genai.load_prompt(f"prompts:/{name}@{ALIAS}",
                                         allow_missing=True, link_to_model=False)
        except Exception:
            # Alias non ancora esistente: è la condizione normale al primo avvio.
            corrente = None

        if corrente is not None and corrente.template == template:
            logger.info("Prompt '%s': invariato, resta la versione %s", name, corrente.version)
            _cache[name] = corrente
            return corrente

        nuova = genai.register_prompt(name=name, template=template,
                                      commit_message=commit_message)
        genai.set_prompt_alias(name, ALIAS, nuova.version)
        logger.info("Prompt '%s': registrata la versione %s", name, nuova.version)
        _cache[name] = nuova
        return nuova

    except Exception:
        logger.warning("Prompt registry non disponibile per '%s': il monitoraggio "
                       "delle versioni è disattivato.", name, exc_info=True)
        _cache[name] = None
        return None


def register_agent_prompt() -> Optional[Any]:
    """Registra il prompt di sistema dell'agente, leggendolo da `app/llm.py`."""
    from app.llm import SYSTEM_PROMPT

    return ensure_registered(
        AGENT_PROMPT_NAME,
        SYSTEM_PROMPT,
        commit_message="Prompt dell'agente di help desk",
    )


def as_params(prefix: str = "prompt") -> Dict[str, Any]:
    """
    Versione dei prompt attivi, in forma loggabile come parametri di un run.

    È il punto che rende confrontabili due valutazioni: affiancando i run si
    vede subito se una differenza nelle metriche coincide con un cambio di
    versione del prompt oppure no.
    """
    params: Dict[str, Any] = {}
    for etichetta, nome in (("agent", AGENT_PROMPT_NAME), ("judge", JUDGE_PROMPT_NAME)):
        versione = _cache.get(nome)
        if versione is not None:
            params[f"{prefix}/{etichetta}_version"] = versione.version
            params[f"{prefix}/{etichetta}_uri"] = versione.uri
    return params
