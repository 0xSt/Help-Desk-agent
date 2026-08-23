"""
evaluation/metrics.py
=====================
Metriche per l'evaluation del sistema. **Funzioni pure**: prendono in ingresso
risultati già raccolti e restituiscono numeri, senza sapere nulla di Qdrant,
di Gemini o di MLflow. Sono quindi testabili senza credenziali né servizi
attivi, ed è il motivo per cui questo file è già completo mentre l'harness che
lo userà (`run_evaluation.py`) è ancora un abbozzo.

Tre famiglie di metriche, corrispondenti alle tre cose che vogliamo misurare
separatamente. Tenerle distinte è essenziale: un'unica "accuracy" complessiva
nasconderebbe il fatto che un retrieval mediocre e una logica di decisione
sbagliata sono problemi diversi, con rimedi diversi.

1. **Retrieval** — i chunk recuperati sono quelli giusti?
2. **Escalation** — la decisione di passare a un umano è corretta?
3. **Risposte** — la risposta finale è fondata e utile? (delegata agli scorer
   LLM-as-judge di MLflow, non implementabile con formule chiuse: vedi le note
   in fondo)

NOTA SULL'ASIMMETRIA DEI COSTI
------------------------------
Per l'escalation la metrica di riferimento **non è l'accuracy**. Con circa il
24% di positivi, un modello che non escala mai raggiungerebbe il 76% di
accuracy pur essendo inutile e pericoloso. E i due errori non pesano uguale:

- **falso negativo** (doveva escalare, non l'ha fatto): un incidente di
  sicurezza non rivisto, un accesso concesso senza approvazione. Grave.
- **falso positivo** (ha escalato senza necessità): un operatore spende
  qualche minuto su un ticket che l'agente avrebbe chiuso. Costoso ma non
  dannoso.

Quindi la metrica primaria è il **recall sulla classe "escalate"**, con la
precision monitorata come costo operativo. `fbeta` con beta=2 riassume le due
pesando il recall il quadruplo della precision.
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set


# ==========================================================================
# Utility condivise
# ==========================================================================

def percentile(valori: Sequence[float], p: float) -> float:
    """
    Percentile per interpolazione al più vicino, con p in [0, 1].

    Definito qui e non duplicato altrove: `calibrate_thresholds.py` importa
    questa funzione, così le statistiche sui punteggi che compaiono nei due
    strumenti sono calcolate allo stesso modo e restano confrontabili.
    """
    if not valori:
        return 0.0
    s = sorted(valori)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


# ==========================================================================
# 1. Metriche di retrieval
# ==========================================================================

def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Quota di documenti rilevanti che compaiono nei primi k risultati.

    Se non esiste alcun documento rilevante per la query, la metrica non è
    definita: restituiamo 0.0 e sta al chiamante escludere quel caso dalla
    media (vedi `aggregate_retrieval`).
    """
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def capped_recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Recall con denominatore limitato a `min(|rilevanti|, k)`.

    Perché serve: con la proxy "stessa sottocategoria è rilevante", una query
    di Password Reset ha 9 documenti rilevanti su 135. Con k=3 un sistema
    **perfetto** può recuperarne al massimo 3, quindi `recall@3` avrebbe un
    tetto di 0,33 e sembrerebbe pessimo pur essendo ottimo. È un artefatto
    della misura, non una proprietà del sistema.

    Questa variante normalizza sul massimo ottenibile, così 1.0 significa
    davvero "ha recuperato tutto ciò che poteva". Resta comunque una metrica
    secondaria: le primarie per kb_tickets sono `hit_rate@k` e `mrr`, che il
    problema non ce l'hanno per costruzione.
    """
    if not relevant or k <= 0:
        return 0.0
    massimo = min(len(relevant), k)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / massimo


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int,
                   ignored: Set[str] = frozenset()) -> float:
    """
    Quota dei primi k risultati che sono effettivamente rilevanti.

    `ignored` contiene risultati **pertinenti ma non indispensabili**, che non
    vanno contati né come successo né come errore: vengono rimossi dal
    denominatore. Serve per la KB delle policy, dove la ground truth
    (`policy_relevance.json`) distingue le policy `expected`, senza le quali
    una risposta corretta non è formulabile, da quelle `acceptable`, che sono
    legittimamente pertinenti. Senza questa esclusione il sistema verrebbe
    penalizzato per aver recuperato POL-006 su un ticket di escalation, che è
    esattamente ciò che dovrebbe fare.
    """
    if k <= 0:
        return 0.0
    top = [d for d in retrieved[:k] if d not in ignored or d in relevant]
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0 se almeno un rilevante compare nei primi k, altrimenti 0.0."""
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """
    1/posizione del primo risultato rilevante (0.0 se non ce n'è nessuno).

    Mediato su tutte le query dà l'MRR. Rispetto a recall@k premia il fatto
    che il risultato giusto stia *in alto*, non solo che sia presente: conta,
    perché passiamo al modello solo i primi k chunk e il primo pesa di più nel
    prompt.
    """
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


@dataclass
class RetrievalCase:
    """Un caso di test per il retrieval."""
    query_id: str
    retrieved_ids: List[str]     # id restituiti, in ordine di ranking
    relevant_ids: Set[str]       # ground truth: attesi
    top_score: float = 0.0       # punteggio del primo risultato
    # Risultati pertinenti ma non indispensabili: esclusi dal denominatore
    # della precision, così non vengono contati come errori.
    ignored_ids: Set[str] = field(default_factory=set)


def aggregate_retrieval(cases: Iterable[RetrievalCase], k: int = 3) -> Dict[str, float]:
    """
    Media le metriche su tutti i casi, ignorando quelli senza ground truth.

    Include anche la distribuzione dei punteggi di similarità: serve a tarare
    `ESCALATION_MIN_RETRIEVAL_SCORE`, che oggi è un valore provvisorio. Senza
    guardare dove cadono davvero i punteggi, quella soglia resta un numero
    scelto a caso.
    """
    usable = [c for c in cases if c.relevant_ids]
    if not usable:
        return {}

    n = len(usable)
    scores = [c.top_score for c in usable]
    return {
        # Primarie: non hanno il tetto artificiale descritto in capped_recall_at_k.
        f"hit_rate@{k}": sum(hit_rate_at_k(c.retrieved_ids, c.relevant_ids, k) for c in usable) / n,
        "mrr": sum(reciprocal_rank(c.retrieved_ids, c.relevant_ids) for c in usable) / n,
        # Secondarie.
        f"capped_recall@{k}": sum(capped_recall_at_k(c.retrieved_ids, c.relevant_ids, k) for c in usable) / n,
        f"recall@{k}": sum(recall_at_k(c.retrieved_ids, c.relevant_ids, k) for c in usable) / n,
        f"precision@{k}": sum(
            precision_at_k(c.retrieved_ids, c.relevant_ids, k, c.ignored_ids) for c in usable
        ) / n,
        "n_cases": float(n),
        # Statistiche sui punteggi, per la taratura delle soglie.
        "top_score_mean": sum(scores) / n,
        "top_score_p10": percentile(scores, 0.10),
        "top_score_median": percentile(scores, 0.50),
    }


# ==========================================================================
# 2. Metriche della decisione di escalation
# ==========================================================================

@dataclass
class ConfusionMatrix:
    """Classe positiva = "il ticket va escalato"."""
    tp: int = 0   # doveva escalare, ha escalato
    fp: int = 0   # non doveva, ha escalato          -> costo operativo
    fn: int = 0   # doveva escalare, NON l'ha fatto  -> errore grave
    tn: int = 0   # non doveva, non l'ha fatto

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        """Metrica primaria: quanti dei ticket da escalare sono stati colti."""
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        """Quota di ticket risolvibili lasciati correttamente all'agente."""
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def accuracy(self) -> float:
        """Riportata per completezza, ma fuorviante su classi sbilanciate."""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    def fbeta(self, beta: float = 2.0) -> float:
        """
        Media armonica pesata. beta=2 pesa il recall quattro volte la
        precision, coerentemente con l'asimmetria dei costi descritta in testa
        al modulo.
        """
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        b2 = beta ** 2
        return (1 + b2) * p * r / (b2 * p + r)

    def as_metrics(self, prefix: str = "escalation") -> Dict[str, float]:
        return {
            f"{prefix}/true_positives": float(self.tp),
            f"{prefix}/false_positives": float(self.fp),
            f"{prefix}/false_negatives": float(self.fn),
            f"{prefix}/true_negatives": float(self.tn),
            f"{prefix}/precision": self.precision,
            f"{prefix}/recall": self.recall,          # <- primaria
            f"{prefix}/specificity": self.specificity,
            f"{prefix}/f2": self.fbeta(2.0),
            f"{prefix}/accuracy": self.accuracy,
        }


def confusion_matrix(predicted: Sequence[bool], expected: Sequence[bool]) -> ConfusionMatrix:
    if len(predicted) != len(expected):
        raise ValueError(
            f"predicted ({len(predicted)}) ed expected ({len(expected)}) hanno lunghezza diversa"
        )
    cm = ConfusionMatrix()
    for p, e in zip(predicted, expected):
        if e and p:
            cm.tp += 1
        elif e and not p:
            cm.fn += 1
        elif not e and p:
            cm.fp += 1
        else:
            cm.tn += 1
    return cm


@dataclass
class EscalationCase:
    """Esito della decisione su un singolo caso, con la ground truth attesa."""
    case_id: str
    predicted_escalate: bool
    expected_escalate: bool
    predicted_trigger_codes: List[str]
    expected_trigger_codes: List[str]
    trigger_family: str = "none"   # mandatory | confidence | retrieval | none


def trigger_accuracy(cases: Iterable[EscalationCase]) -> Dict[str, float]:
    """
    Accuratezza **per singolo trigger**, non solo sulla decisione complessiva.

    È il vantaggio concreto di aver fatto restituire a `escalation.decide()`
    una lista di trigger anziché un booleano: quando il recall cala si può
    vedere *quale* segnale non sta scattando, invece di sapere solo che il
    sistema sbaglia. Per ogni codice di policy atteso misuriamo quante volte
    la decisione lo ha effettivamente prodotto.
    """
    attesi: Counter = Counter()
    colti: Counter = Counter()
    for c in cases:
        pred = set(c.predicted_trigger_codes)
        for code in set(c.expected_trigger_codes):
            attesi[code] += 1
            if code in pred:
                colti[code] += 1
    return {
        f"trigger/{code}/recall": colti[code] / attesi[code]
        for code in sorted(attesi) if attesi[code]
    }


def per_family_breakdown(cases: Iterable[EscalationCase]) -> Dict[str, float]:
    """
    Recall separato per famiglia di segnale.

    Distingue i due regimi: i casi `mandatory` dipendono dalla classificazione
    del modello e da regole fisse, quelli `confidence`/`retrieval` dipendono
    dalle soglie configurate. Un calo nelle prime e uno nelle seconde si
    correggono in modi completamente diversi.
    """
    per_family: Dict[str, List[EscalationCase]] = {}
    for c in cases:
        if c.expected_escalate:
            per_family.setdefault(c.trigger_family, []).append(c)

    out = {}
    for family, group in sorted(per_family.items()):
        colti = sum(1 for c in group if c.predicted_escalate)
        out[f"family/{family}/recall"] = colti / len(group)
        out[f"family/{family}/n"] = float(len(group))
    return out


def evaluate_escalation(cases: Sequence[EscalationCase]) -> Dict[str, float]:
    """Tutte le metriche di escalation in un unico dizionario loggabile."""
    cm = confusion_matrix(
        [c.predicted_escalate for c in cases],
        [c.expected_escalate for c in cases],
    )
    metrics = cm.as_metrics()
    metrics.update(trigger_accuracy(cases))
    metrics.update(per_family_breakdown(cases))
    metrics["escalation/n_cases"] = float(len(cases))
    return metrics


# ==========================================================================
# 3. Qualità delle risposte — nota di progetto
# ==========================================================================
#
# Non implementata qui, e deliberatamente: "la risposta è corretta e fondata
# sul contesto recuperato?" non si calcola con una formula chiusa. Le opzioni
# sono LLM-as-judge o annotazione manuale.
#
# MLflow espone scorer già pronti utilizzabili via `mlflow.genai.evaluate()`:
#   - RetrievalGroundedness  -> la risposta è supportata dal contesto recuperato?
#   - RetrievalSufficiency   -> il contesto recuperato basta a rispondere?
#   - RelevanceToQuery       -> la risposta affronta davvero la domanda?
#   - Correctness            -> confronto con una risposta di riferimento
#
# Per il nostro dominio serve almeno uno scorer custom: **policy compliance**,
# cioè "la risposta rispetta i vincoli della policy citata?" — per esempio non
# promette l'esito di un'approvazione, non condivide credenziali su canali non
# verificati, non chiude un ticket di sicurezza per conto suo.
#
# Come riferimento per la Correctness abbiamo `resolution_summary` dei ticket
# storici. Attenzione al leakage: se il ticket usato come query è anche
# indicizzato in kb_tickets, il retrieval trova sé stesso e la risposta è
# banalmente corretta. Va usata la modalità leave-one-out.


# ==========================================================================
# Utility
# ==========================================================================

def sanitize_metric_name(name: str) -> str:
    """
    Rende un nome di metrica accettabile da MLflow.

    MLflow ammette solo lettere, cifre, underscore, trattini, punti, spazi,
    barre e due punti. I nostri nomi contengono i codici di policy, e il
    carattere di sezione (§) non è fra quelli ammessi: `log_metrics` rifiuta
    **l'intero batch** se anche un solo nome è invalido, quindi senza questa
    normalizzazione andrebbero perse *tutte* le metriche, non solo quelle con
    il carattere incriminato.

    La sostituzione mantiene leggibile la corrispondenza con la clausola di
    origine: "trigger/POL-006 §3.1/recall" diventa
    "trigger/POL-006_sec3.1/recall".
    """
    pulito = name.replace("§", "sec").replace(" ", "_")
    return "".join(c for c in pulito if c.isalnum() or c in "_-./: ")


def flatten_metrics(*groups: Dict[str, float]) -> Dict[str, float]:
    """
    Unisce più dizionari di metriche, normalizzandone i nomi per MLflow.

    La normalizzazione avviene qui, in un unico punto attraversato da tutte le
    metriche prima di essere registrate, invece che nei singoli produttori:
    così le funzioni di misura restano libere di usare i nomi naturali del
    dominio, con i codici di policy scritti per esteso.
    """
    out: Dict[str, float] = {}
    for g in groups:
        for k, v in g.items():
            out[sanitize_metric_name(k)] = v
    return out
