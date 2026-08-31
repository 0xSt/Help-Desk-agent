"""
eval_suite/metrics.py
=====================
Metriche per la decisione di escalation. **Funzioni pure**: ricevono esiti già
raccolti e restituiscono numeri, senza toccare MLflow, Qdrant o il modello.
Sono quindi verificabili senza credenziali né servizi attivi.

Quali metriche, e perché queste
-------------------------------
La classe positiva è "il ticket va escalato". La scelta non è neutra, perché i
due errori hanno costi molto diversi:

- **falso negativo** — doveva escalare e non l'ha fatto: un incidente di
  sicurezza mai rivisto, un accesso concesso senza approvazione;
- **falso positivo** — ha escalato inutilmente: qualche minuto di un operatore.

Da qui la gerarchia adottata:

| Metrica | Ruolo | Perché |
|---|---|---|
| `recall` | **primaria** | quota di ticket da escalare effettivamente colti: è l'errore che costa |
| `precision` | costo operativo | quanta parte delle escalation era superflua |
| `f2` | sintesi | media armonica che pesa il richiamo quattro volte la precisione, coerente con l'asimmetria |
| `mcc` | sintesi robusta | coefficiente di correlazione di Matthews |
| `specificity` | complemento | quota di ticket risolvibili lasciati correttamente all'agente |
| conteggi `tp/fp/fn/tn` | diagnosi | permettono di ricostruire qualunque altra misura e di capire se un valore anomalo dipende da pochi casi |

**L'accuratezza non compare fra le primarie**, deliberatamente: con circa un
quarto di positivi, un sistema che non escala mai raggiungerebbe il 76% pur
essendo inutile e pericoloso.

**Perché anche l'MCC.** F2 ignora i veri negativi, quindi non distingue un
sistema che lascia correttamente passare i ticket risolvibili da uno che ne
escala molti senza necessità, purché il richiamo resti alto. L'MCC usa tutte e
quattro le celle della matrice e resta informativo con classi sbilanciate: vale
+1 per una previsione perfetta, 0 per una previsione casuale, −1 per una
sistematicamente invertita. Serve da controprova sintetica dell'F2, non a
sostituirlo.
"""
from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Sequence


@dataclass
class Esito:
    """Esito della decisione su un singolo caso, con l'attesa corrispondente."""
    case_id: str
    predetto: bool
    atteso: bool
    codici_predetti: List[str]
    codici_attesi: List[str]
    famiglia: str = "none"


def _matrice(esiti: Sequence[Esito]) -> Dict[str, int]:
    """Conteggi della matrice di confusione, con positivo = «va escalato»."""
    m = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for e in esiti:
        if e.atteso and e.predetto:
            m["tp"] += 1
        elif e.atteso and not e.predetto:
            m["fn"] += 1
        elif not e.atteso and e.predetto:
            m["fp"] += 1
        else:
            m["tn"] += 1
    return m


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """
    Coefficiente di correlazione di Matthews.

    Il denominatore si annulla quando una riga o una colonna della matrice è
    interamente nulla, cioè quando il sistema predice una sola classe oppure il
    dataset ne contiene una sola. In quei casi la correlazione non è definita e
    si restituisce 0.0, che è anche il valore corrispondente a una previsione
    non informativa: è la convenzione usuale ed evita di far apparire come
    eccellente un sistema degenere.
    """
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        return 0.0
    return (tp * tn - fp * fn) / den


def aggrega(esiti: Sequence[Esito]) -> Dict[str, float]:
    """Tutte le metriche di escalation, pronte per essere registrate."""
    m = _matrice(esiti)
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if (precision + recall) else 0.0

    metriche = {
        "escalation/recall": recall,
        "escalation/precision": precision,
        "escalation/f2": f2,
        "escalation/mcc": _mcc(tp, fp, fn, tn),
        "escalation/specificity": specificity,
        "escalation/accuracy": (tp + tn) / len(esiti) if esiti else 0.0,
        "escalation/n_cases": float(len(esiti)),
        **{f"escalation/{k}": float(v) for k, v in m.items()},
    }
    metriche.update(_per_clausola(esiti))
    metriche.update(_per_famiglia(esiti))
    return metriche


def _per_clausola(esiti: Iterable[Esito]) -> Dict[str, float]:
    """
    Richiamo per singola clausola di policy.

    È l'informazione che un richiamo aggregato non può dare: distingue un
    sistema che sbaglia in modo diffuso da uno che ignora sistematicamente una
    regola precisa. Cattura anche il caso di una decisione corretta presa per il
    motivo sbagliato, che nella matrice di confusione conta come successo.
    """
    attesi: Dict[str, int] = {}
    colti: Dict[str, int] = {}
    for e in esiti:
        predetti = set(e.codici_predetti)
        for codice in set(e.codici_attesi):
            attesi[codice] = attesi.get(codice, 0) + 1
            if codice in predetti:
                colti[codice] = colti.get(codice, 0) + 1
    return {f"clausola/{_pulisci(c)}/recall": colti.get(c, 0) / n
            for c, n in sorted(attesi.items()) if n}


def _per_famiglia(esiti: Iterable[Esito]) -> Dict[str, float]:
    """
    Richiamo per famiglia di segnale, sui soli casi che devono escalare.

    Separa due regimi che si correggono in modi diversi: i casi `mandatory`
    dipendono da quanto bene il modello estrae i segnali dal testo, quelli
    `confidence` e `retrieval` dalle soglie configurate. Un calo nei primi si
    affronta sul prompt, nei secondi sui numeri.
    """
    gruppi: Dict[str, List[Esito]] = {}
    for e in esiti:
        if e.atteso:
            gruppi.setdefault(e.famiglia, []).append(e)
    out: Dict[str, float] = {}
    for famiglia, casi in sorted(gruppi.items()):
        out[f"famiglia/{famiglia}/recall"] = sum(1 for c in casi if c.predetto) / len(casi)
        out[f"famiglia/{famiglia}/n"] = float(len(casi))
    return out


def _pulisci(nome: str) -> str:
    """
    Rende un nome accettabile da MLflow come chiave di metrica.

    MLflow ammette lettere, cifre, underscore, trattini, punti, spazi, barre e
    due punti. I codici di policy contengono il segno di paragrafo, che non è
    fra questi, e `log_metrics` rifiuta **l'intero blocco** se anche un solo
    nome è invalido: senza normalizzazione andrebbero perse tutte le metriche,
    non solo quella con il carattere incriminato.
    """
    return nome.replace("§", "sec").replace(" ", "_")
