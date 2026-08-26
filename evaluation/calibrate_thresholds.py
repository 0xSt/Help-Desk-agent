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
- **fuori dominio**: 70 query plausibili ma su argomenti che nessuna policy
  tratta (`out_of_domain_queries.json`), distribuite su otto aree tematiche.
  La varietà conta quanto la numerosità: un campione concentrato su un solo
  tipo di estraneità darebbe una stima valida solo per quel tipo. Sono incluse
  di proposito query lessicalmente vicine al dominio — domande tecniche non
  coperte da policy, e domande che citano l'IT senza chiedere supporto — perché
  sono i casi che una soglia di similarità separa con più difficoltà, e
  ometterli renderebbe la stima ottimistica.

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
from typing import Any, Dict, List, Sequence, Tuple

from app import config
from app.retrieval import search_kb_docs, search_kb_tickets
# Percentile condiviso con metrics.py: le statistiche sui punteggi prodotte
# dai due strumenti devono essere calcolate allo stesso modo.
from evaluation.metrics import percentile as _percentile

logging.basicConfig(level=logging.WARNING)

DATASETS = Path(__file__).parent / "datasets"
KB = Path(__file__).parent.parent / "app" / "knowledge_base"


def _riassunto(nome: str, valori: List[float]) -> Dict[str, Any]:
    """
    Statistiche descrittive di una popolazione di punteggi.

    Si riportano i percentili e non media e deviazione standard perché la
    distribuzione dei punteggi di similarità non è simmetrica e non c'è motivo
    di assumerla normale: i percentili descrivono dove cadono davvero i valori
    senza presupporre una forma.

    Servono a leggere la **sovrapposizione** fra le due popolazioni, che è
    l'informazione decisiva: se il p05 delle query in dominio è sotto il p95
    di quelle fuori dominio, esiste una fascia in cui nessuna soglia separa
    correttamente, e la scelta si riduce a decidere quale dei due errori
    tollerare.
    """
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
    """
    Calcola il miglior punteggio di similarità per ciascuna delle due
    popolazioni: query in dominio e query fuori dominio.

    Due dettagli che ne determinano la validità.

    **Leave-one-out sulle query in dominio.** Ogni ticket usato come query è
    anche indicizzato: senza escluderlo recupererebbe sé stesso con similarità
    prossima a 1, e la distribuzione in dominio risulterebbe artificialmente
    alta. La soglia che ne deriverebbe sarebbe tarata su un fenomeno che in
    esercizio non si verifica mai.

    **Campionamento riproducibile.** Il seme è fisso perché due esecuzioni
    consecutive devono proporre la stessa soglia: se il campione cambiasse a
    ogni lancio, la variazione del valore proposto sarebbe rumore
    indistinguibile da un effetto reale.

    Il parametro `campione` esiste per contenere il costo durante lo sviluppo:
    ogni query in dominio costa un embedding, e 135 embedding per ogni prova
    di taratura sono superflui quando serve solo un ordine di grandezza.
    """
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
    """
    Esegue la calibrazione e stampa un riepilogo leggibile.

    Non modifica alcun file: **propone** un valore e lascia all'operatore la
    decisione di adottarlo. È deliberato — la soglia è un parametro che incide
    sul comportamento del sistema in produzione, e una modifica automatica
    partendo da un campione ridotto o da un indice costruito col provider di
    fallback produrrebbe un cambiamento silenzioso e potenzialmente sbagliato.

    Quando la separazione fra le due popolazioni è debole lo dichiara
    esplicitamente: in quel caso il problema non è la taratura ma la qualità
    del recupero, e adottare comunque il valore proposto darebbe l'illusione
    di aver risolto qualcosa.
    """
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
    print(f"  fuori dominio nel range di quelle in dominio: "
          f"{esito['ood_oltre_p05_in_dominio']:.0f} su {len(fuori_dominio)} "
          f"(sovrapposizione irriducibile fra le due popolazioni)")

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
