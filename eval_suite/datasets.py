"""
eval_suite/datasets.py
======================
Caricamento dei dataset di valutazione, a partire dai dati **già presenti** nel
progetto: non viene generato nulla di nuovo.

Due dataset, per due domande diverse.

`escalation_cases()` — 43 casi scritti a mano, con l'esito atteso annotato.
Servono a misurare la **decisione**: sono l'unico materiale che copre anche i
criteri di POL-006 §4 (bassa confidenza, retrieval senza appigli) e i redirect
fuori ambito, che lo storico dei ticket non contiene, essendo stato gestito
interamente da operatori umani.

`retrieval_cases()` — i ticket storici usati come interrogazioni. Servono a
misurare la **qualità del contesto** e della risposta: sono richieste reali,
formulate come le scriverebbe un utente, e la knowledge base le copre per
costruzione.

La separazione è deliberata: mescolarli produrrebbe una media che nasconde
quale dei due aspetti sta funzionando male.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List

KB = Path(__file__).resolve().parent.parent / "app" / "knowledge_base"
# I dataset vivono accanto al codice che li usa. La cartella si chiama `data`
# e non `datasets` perché in questo package esiste già il modulo
# `datasets.py`: una directory omonima renderebbe ambiguo l'import.
DATI = Path(__file__).resolve().parent / "data"


def escalation_cases() -> List[Dict[str, Any]]:
    """
    Casi etichettati per la decisione di escalation, nel formato di MLflow.

    Ogni riga porta `inputs` (ciò che il sistema riceve) ed `expectations` (la
    verità attesa). Fra le attese c'è anche l'elenco delle clausole di policy
    che dovrebbero scattare: consente di misurare non solo *se* la decisione è
    corretta, ma *per quale motivo*, distinguendo una decisione giusta presa
    per la ragione sbagliata.
    """
    casi = json.loads((DATI / "escalation_cases.json").read_text(encoding="utf-8"))
    return [
        {
            "inputs": {"query": c["query"]},
            "expectations": {
                "escalate": c["expected_escalate"],
                "trigger_codes": c["expected_trigger_codes"],
            },
            "case_id": c["case_id"],
            "trigger_family": c["trigger_family"],
        }
        for c in casi
    ]


def retrieval_cases(sample: int = 0, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Ticket storici usati come interrogazioni, per la qualità di contesto e
    risposta.

    `ticket_id` viaggia negli `inputs` perché serve al leave-one-out: il ticket
    impiegato come interrogazione va escluso dai risultati, altrimenti si
    recupera da solo con similarità prossima a 1 e ogni metrica di recupero
    risulta perfetta per costruzione.

    Il campionamento è **casuale semplice e con seme fisso**. Semplice perché
    qui interessa una stima non distorta della qualità media, e stratificare
    per sottocategoria imporrebbe almeno un caso per ciascuna delle venti,
    fissando di fatto un campione minimo di venti richieste anche quando ne
    bastano meno. Con seme fisso perché due esecuzioni consecutive devono
    valutare gli stessi casi: altrimenti la differenza fra due run sarebbe
    rumore campionario indistinguibile da un effetto reale.
    """
    tickets = json.loads((KB / "past_tickets.json").read_text(encoding="utf-8"))
    if sample and sample < len(tickets):
        tickets = random.Random(seed).sample(tickets, sample)

    return [
        {
            "inputs": {
                "query": f"{t['subject']}\n\n{t['description']}",
                "ticket_id": t["ticket_id"],
            },
            "case_id": t["ticket_id"],
            "subcategory": t["subcategory"],
        }
        for t in tickets
    ]
