"""
evaluation/calibrate_thresholds.py
==================================
Calibrazione di `ESCALATION_MIN_RETRIEVAL_SCORE`, la soglia sotto la quale il
retrieval è considerato privo di appigli e scatta l'escalation per POL-006 §4.

    python -m evaluation.calibrate_thresholds

Il problema che risolve
-----------------------
La soglia non è trasferibile tra provider di embedding: i punteggi di
similarità di Gemini e quelli dell'hashing di fallback vivono su scale
diverse e non confrontabili. Va quindi rimisurata ogni volta che si cambia
modello o dimensione dei vettori.

Il metodo, e perché non usa i casi di test
------------------------------------------
Si confrontano le distribuzioni dei punteggi di due popolazioni:

- **in dominio**: i ticket storici usati come query, in leave-one-out. Sono
  per definizione richieste che la knowledge base copre.
- **fuori dominio**: query plausibili ma su argomenti che nessuna policy
  tratta (`out_of_domain_queries.json`).

**Nessuna delle due richiede etichette di escalation.** È deliberato: tarare
la soglia guardando i 23 casi etichettati e poi misurare il sistema su quegli
stessi casi significherebbe adattare i parametri al test set, e i risultati
non direbbero più nulla. Così i casi etichettati restano intatti.

La soglia proposta è il punto che separa meglio le due distribuzioni. Se si
sovrappongono ampiamente, nessuna soglia funziona bene: è un'informazione
sulla qualità del retrieval, non un problema di taratura.
"""
import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from app import config
from app.retrieval import search_kb_docs, search_kb_tickets

logging.basicConfig(level=logging.WARNING)

DATASETS = Path(__file__).parent / "datasets"
KB = Path(__file__).parent.parent / "app" / "knowledge_base"


def _percentile(valori: Sequence[float], p: float) -> float:
    if not valori:
        return 0.0
    s = sorted(valori)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def _riassunto(nome: str, valori: List[float]) -> Dict[str, float]:
    return {
        "popolazione": nome,
        "n": len(valori),
        "min": min(valori) if valori else 0.0,
        "p05": _percentile(valori, 0.05),
        "p25": _percentile(valori, 0.25),
        "mediana": _percentile(valori, 0.50),
        "p75": _percentile(valori, 0.75),
        "p95": _percentile(valori, 0.95),
        "max": max(valori) if valori else 0.0,
    }


def _best_score(query: str, escludi: Sequence[str] = ()) -> float:
    """
    Miglior punteggio ottenuto su **una qualsiasi** delle due KB.

    Rispecchia esattamente il trigger di grounding implementato in
    `escalation.py`, che scatta solo se *entrambe* le KB restano sotto
    soglia: quindi ciò che conta è il massimo tra le due.
    """
    docs = search_kb_docs(query, k=1)
    tickets = search_kb_tickets(query, k=1, exclude_sources=escludi)
    punteggi = [r["score"] for r in docs + tickets]
    return max(punteggi) if punteggi else 0.0


def raccogli_punteggi(campione: int = 0) -> Tuple[List[float], List[float]]:
    tickets = json.loads((KB / "past_tickets.json").read_text(encoding="utf-8"))
    if campione and campione < len(tickets):
        random.seed(42)  # riproducibile: due esecuzioni danno la stessa soglia
        tickets = random.sample(tickets, campione)

    in_dominio = []
    for t in tickets:
        query = f"{t['subject']}\n\n{t['description']}"
        # Leave-one-out: senza, il ticket recupererebbe sé stesso con
        # similarità quasi 1.0 e la distribuzione in dominio sarebbe finta.
        in_dominio.append(_best_score(query, escludi=[t["ticket_id"]]))

    ood_file = json.loads((DATASETS / "out_of_domain_queries.json").read_text(encoding="utf-8"))
    fuori_dominio = [_best_score(q) for q in ood_file["queries"]]

    return in_dominio, fuori_dominio


def proponi_soglia(in_dominio: List[float], fuori_dominio: List[float]) -> Dict[str, float]:
    """
    Cerca la soglia che massimizza la separazione tra le due popolazioni.

    Il criterio è la **media tra le due accuratezze** (quota di in dominio
    correttamente sopra soglia e quota di fuori dominio correttamente sotto),
    non l'accuratezza semplice: le due popolazioni hanno numerosità molto
    diverse (135 contro 20) e l'accuratezza semplice sarebbe dominata dalla
    più numerosa, producendo una soglia che ignora i fuori dominio.
    """
    candidati = sorted(set(round(v, 3) for v in in_dominio + fuori_dominio))
    migliore, miglior_punteggio = 0.0, -1.0

    for soglia in candidati:
        sopra = sum(1 for v in in_dominio if v >= soglia) / max(len(in_dominio), 1)
        sotto = sum(1 for v in fuori_dominio if v < soglia) / max(len(fuori_dominio), 1)
        punteggio = (sopra + sotto) / 2
        if punteggio > miglior_punteggio:
            migliore, miglior_punteggio = soglia, punteggio

    sopra = sum(1 for v in in_dominio if v >= migliore) / max(len(in_dominio), 1)
    sotto = sum(1 for v in fuori_dominio if v < migliore) / max(len(fuori_dominio), 1)
    sovrapposizione = sum(1 for v in fuori_dominio if v >= _percentile(in_dominio, 0.05))
    return {
        "soglia_proposta": migliore,
        "separazione": miglior_punteggio,
        "in_dominio_sopra_soglia": sopra,
        "fuori_dominio_sotto_soglia": sotto,
        "ood_oltre_p05_in_dominio": sovrapposizione,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrazione delle soglie di retrieval")
    parser.add_argument("--sample", type=int, default=0,
                        help="usa solo N ticket in dominio (0 = tutti)")
    args = parser.parse_args()

    print("=" * 70)
    print("CALIBRAZIONE DELLA SOGLIA DI GROUNDING")
    print("=" * 70)
    print(f"  provider embedding : {config.embedding_provider()}")
    print(f"  modello            : {config.GEMINI_EMBEDDING_MODEL}")
    print(f"  dimensione         : {config.active_embedding_dim()}")
    print(f"  soglia attuale     : {config.MIN_RETRIEVAL_SCORE}")
    if config.embedding_provider() != "gemini":
        print("\n  ATTENZIONE: provider di fallback attivo. I punteggi non sono")
        print("  rappresentativi e la soglia che ne esce non va usata.")

    in_dominio, fuori_dominio = raccogli_punteggi(args.sample)

    print("\nDISTRIBUZIONE DEL MIGLIOR PUNTEGGIO PER QUERY\n")
    intestazione = f"  {'popolazione':16s} {'n':>4} {'min':>7} {'p05':>7} {'p25':>7} {'mediana':>8} {'p75':>7} {'p95':>7} {'max':>7}"
    print(intestazione)
    print("  " + "-" * (len(intestazione) - 2))
    for nome, valori in (("in dominio", in_dominio), ("fuori dominio", fuori_dominio)):
        r = _riassunto(nome, valori)
        print(f"  {r['popolazione']:16s} {r['n']:4d} {r['min']:7.3f} {r['p05']:7.3f} "
              f"{r['p25']:7.3f} {r['mediana']:8.3f} {r['p75']:7.3f} {r['p95']:7.3f} {r['max']:7.3f}")

    esito = proponi_soglia(in_dominio, fuori_dominio)

    print("\nSOGLIA PROPOSTA\n")
    print(f"  ESCALATION_MIN_RETRIEVAL_SCORE = {esito['soglia_proposta']:.3f}")
    print(f"  separazione media              : {esito['separazione']:.1%}")
    print(f"  query in dominio sopra soglia  : {esito['in_dominio_sopra_soglia']:.1%} "
          f"(quelle sotto verrebbero escalate per mancanza di appigli)")
    print(f"  query fuori dominio sotto      : {esito['fuori_dominio_sotto_soglia']:.1%} "
          f"(quelle sopra NON verrebbero intercettate)")

    if esito["separazione"] < 0.75:
        print("\n  Separazione debole: le due distribuzioni si sovrappongono molto.")
        print("  Nessuna soglia darà buoni risultati; il problema è la qualità del")
        print("  retrieval, non la taratura. Da rivedere prima di procedere.")
    else:
        print(f"\n  Scrivi il valore in .env, poi rilancia l'ingestion non serve:")
        print(f"  la soglia è usata a runtime, basta riavviare il backend.")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
