"""
analysis/eda.py
===============
Exploratory data analysis sulle due knowledge base del progetto.

Obiettivi (in ordine di importanza per il progetto):

1. **Categorizzare i dati**: capire come sono distribuiti i ticket per
   categoria/sottocategoria/priorità, e quali sono le dimensioni utili a
   filtrare o segmentare il retrieval in futuro.
2. **Preparare l'estensione della KB**: sapere esattamente quale
   distribuzione replicare quando si generano ticket sintetici aggiuntivi.
3. **Progettare i dataset di evaluation**: quantificare la ground truth
   disponibile per l'escalation (`was_escalated_to_human`), verificarne la
   coerenza con le policy, e misurare lo sbilanciamento delle classi.
4. **Vincoli tecnici per l'embedding**: lunghezza dei testi da embeddare,
   per verificare che stiano nei limiti del modello di embedding scelto
   (gemini-embedding-001 accetta fino a 2048 token per input).

Esecuzione:
    python analysis/eda.py

Produce:
    analysis/eda_report.md      report testuale completo
    analysis/figures/*.png      grafici delle distribuzioni principali
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend non interattivo: scriviamo su file, non a schermo
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "app" / "knowledge_base"
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

# Stima grossolana token ~= parole * 1.3 (sufficiente per verificare che siamo
# lontanissimi dal limite di 2048 token, non serve un tokenizer vero).
TOKENS_PER_WORD = 1.3


def load_tickets():
    """Carica lo storico ticket. Corpus unico: tutti i dati sono simulati."""
    return json.loads((KB_DIR / "past_tickets.json").read_text(encoding="utf-8"))


def load_policy_sections():
    """Spezza le policy per sezione '## ', come fa app/retrieval.py."""
    sections = []
    for md_path in sorted((KB_DIR / "policies").glob("*.md")):
        lines = md_path.read_text(encoding="utf-8").splitlines()
        doc_title = lines[0].lstrip("# ").strip() if lines else md_path.stem
        current_title, current_lines = None, []

        def flush():
            if current_title and current_lines:
                body = "\n".join(current_lines).strip()
                sections.append({
                    "file": md_path.name,
                    "policy_id": md_path.stem.split("-")[0] + "-" + md_path.stem.split("-")[1],
                    "doc_title": doc_title,
                    "section_title": current_title,
                    "text": f"{doc_title} — {current_title}\n\n{body}",
                })

        for line in lines:
            if line.startswith("## "):
                flush()
                current_title, current_lines = line[3:].strip(), []
            elif current_title is not None:
                current_lines.append(line)
        flush()
    return sections


def fmt_table(headers, rows):
    """Tabella Markdown."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def bar_chart(counter, title, filename, xlabel="", rotate=30, color="#3b4a6b"):
    labels = [k for k, _ in counter.most_common()]
    values = [v for _, v in counter.most_common()]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel("n. ticket")
    if xlabel:
        ax.set_xlabel(xlabel)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def stacked_escalation_chart(tickets, filename):
    cats = [c for c, _ in Counter(t["category"] for t in tickets).most_common()]
    esc = [sum(1 for t in tickets if t["category"] == c and t["was_escalated_to_human"]) for c in cats]
    noesc = [sum(1 for t in tickets if t["category"] == c and not t["was_escalated_to_human"]) for c in cats]

    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 1.1), 4.2))
    ax.bar(cats, noesc, label="risolti dall'agente", color="#c9cfdd")
    ax.bar(cats, esc, bottom=noesc, label="escalati a umano", color="#a8710a")
    ax.set_title("Escalation per categoria")
    ax.set_ylabel("n. ticket")
    ax.legend(frameon=False)
    plt.xticks(rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=120)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tickets = load_tickets()
    sections = load_policy_sections()
    n = len(tickets)

    L = []  # righe del report
    A = L.append

    A("# EDA — knowledge base help desk\n")
    A("> Generato da `analysis/eda.py`. Rigenerabile con `python analysis/eda.py`.\n")

    # ---------------------------------------------------------------- KB 1
    A("## 1. KB ticket storici\n")
    A(f"**{n} ticket** in `past_tickets.json`. L'intero corpus è materiale "
      "simulato costruito per questo progetto: non esiste distinzione tra dati "
      "reali e sintetici, sono tutti dati di scenario trattati allo stesso modo.\n")

    A("### 1.1 Schema\n")
    all_keys = sorted({k for t in tickets for k in t if not k.startswith("_")})
    A("Campi presenti in ogni record: " + ", ".join(f"`{k}`" for k in all_keys) + ".\n")
    A("Il campo chiave per questo progetto è **`was_escalated_to_human`**: è la "
      "ground truth con cui misureremo l'accuratezza della decisione di escalation. "
      "`escalation_reason` ne spiega il motivo ed è utile per capire *quale* regola "
      "di policy ha innescato l'escalation.\n")

    # Distribuzioni
    A("### 1.2 Distribuzione per categoria\n")
    cat_counter = Counter(t["category"] for t in tickets)
    rows = []
    for cat, c in cat_counter.most_common():
        esc = sum(1 for t in tickets if t["category"] == cat and t["was_escalated_to_human"])
        rows.append([cat, c, f"{c / n:.1%}", esc, f"{esc / c:.0%}"])
    A(fmt_table(["Categoria", "N", "% sul totale", "Escalati", "% escalati"], rows) + "\n")
    bar_chart(cat_counter, "Ticket per categoria", "categorie.png")
    A("![Ticket per categoria](figures/categorie.png)\n")

    A("### 1.3 Distribuzione per sottocategoria\n")
    sub_rows = []
    for cat, _ in cat_counter.most_common():
        subs = Counter(t["subcategory"] for t in tickets if t["category"] == cat)
        for sub, c in subs.most_common():
            esc = sum(1 for t in tickets
                      if t["subcategory"] == sub and t["category"] == cat and t["was_escalated_to_human"])
            sub_rows.append([cat, sub, c, esc])
    A(fmt_table(["Categoria", "Sottocategoria", "N", "Escalati"], sub_rows) + "\n")
    A(f"In totale **{len({t['subcategory'] for t in tickets})} sottocategorie** distinte: "
      "una granularità utile come possibile filtro sui payload Qdrant, o come "
      "etichetta da far predire al modello per abilitare regole di escalation "
      "specifiche per dominio.\n")

    A("### 1.4 Priorità, stato, canale\n")
    prio = Counter(t["priority"] for t in tickets)
    A("**Priorità** — " + ", ".join(f"`{k}`: {v}" for k, v in sorted(prio.items())) + "\n")
    A("**Stato** — " + ", ".join(f"`{k}`: {v}" for k, v in Counter(t["status"] for t in tickets).most_common()) + "\n")
    A("**Canale** — " + ", ".join(f"`{k}`: {v}" for k, v in Counter(t["source_channel"] for t in tickets).most_common()) + "\n")
    A("**Reparto** — " + ", ".join(f"`{k}`: {v}" for k, v in Counter(t["department"] for t in tickets).most_common()) + "\n")
    A("**Ruolo richiedente** — " + ", ".join(f"`{k}`: {v}" for k, v in Counter(t["requester_role"] for t in tickets).most_common()) + "\n")
    bar_chart(prio, "Ticket per priorità", "priorita.png", rotate=0, color="#a8710a")
    A("![Ticket per priorità](figures/priorita.png)\n")

    # ------------------------------------------------------- escalation
    A("## 2. Escalation: la ground truth per l'evaluation\n")
    n_esc = sum(1 for t in tickets if t["was_escalated_to_human"])
    A(f"**{n_esc} / {n} ticket escalati ({n_esc / n:.1%})**. "
      f"Classe positiva minoritaria con rapporto ~1:{(n - n_esc) / max(n_esc, 1):.1f}.\n")
    stacked_escalation_chart(tickets, "escalation_per_categoria.png")
    A("![Escalation per categoria](figures/escalation_per_categoria.png)\n")

    A("### 2.1 Escalation per priorità\n")
    rows = []
    for p in sorted(prio):
        tot = prio[p]
        esc = sum(1 for t in tickets if t["priority"] == p and t["was_escalated_to_human"])
        rows.append([p, tot, esc, f"{esc / tot:.0%}"])
    A(fmt_table(["Priorità", "N", "Escalati", "% escalati"], rows) + "\n")

    A("### 2.2 A quale regola di policy corrisponde ogni escalation\n")
    A("POL-006 §3 elenca 8 trigger di escalation obbligatoria. Mappando ogni "
      "`escalation_reason` sul trigger corrispondente si vede quanto la ground "
      "truth sia spiegabile da regole deterministiche:\n")

    # euristica di mappatura sui motivi testuali (solo per il report)
    def map_trigger(t):
        if t["category"] == "Security":
            return "§3.1 security-classified"
        r = (t["escalation_reason"] or "").lower()
        if "termination" in r:
            return "§3.2 involuntary termination"
        if "approval" in r and ("system-owner" in r or "system owner" in r):
            return "§3.3 sensitive access, dual approval"
        if "threshold" in r or "budget" in r or "catalog" in r:
            return "§3.4 spend threshold / non-catalog"
        if "multiple users" in r or "outage" in r:
            return "§3.7 multi-user / infrastructure"
        return "non mappato"

    trig = Counter(map_trigger(t) for t in tickets if t["was_escalated_to_human"])
    A(fmt_table(["Trigger POL-006", "N ticket"], [[k, v] for k, v in trig.most_common()]) + "\n")
    unmapped = [t["ticket_id"] for t in tickets if t["was_escalated_to_human"] and map_trigger(t) == "non mappato"]
    A(f"Ticket escalati non riconducibili a un trigger §3: **{len(unmapped)}**"
      + (f" ({', '.join(unmapped)})" if unmapped else "") + ".\n")

    A("> **Implicazione per l'evaluation.** Le etichette storiche sono quasi "
      "interamente spiegate dai trigger *deterministici* di POL-006 §3. Nessun "
      "ticket è etichettato come escalato per i criteri di POL-006 §4 (bassa "
      "confidenza del modello, retrieval senza risultati sopra soglia): è atteso, "
      "perché sono ticket storici gestiti da umani, non da un agente AI. "
      "Ne segue che **questo dataset da solo non può misurare la parte §4 della "
      "logica di escalation**: servirà un secondo set di casi costruito ad hoc.\n")

    A("### 2.3 Coerenza con le policy\n")
    sec = [t for t in tickets if t["category"] == "Security"]
    sec_not_esc = [t["ticket_id"] for t in sec if not t["was_escalated_to_human"]]
    A(f"POL-005 §8 impone escalation per **ogni** ticket Security. "
      f"Ticket Security presenti: {len(sec)}, di cui non escalati: "
      f"**{len(sec_not_esc)}**" + (f" ({', '.join(sec_not_esc)})" if sec_not_esc else " ✓ coerente") + ".\n")

    # ------------------------------------------------------ testi/embedding
    A("## 3. Caratteristiche dei testi (vincoli per l'embedding)\n")
    A("Il testo che viene effettivamente embeddato per un ticket è il *lato "
      "problema* (`subject` + `description`), perché è ciò a cui somiglia una "
      "nuova richiesta in ingresso; la risoluzione finisce nel payload.\n")

    def wstats(values):
        s = sorted(values)
        return s[0], s[len(s) // 2], s[-1], sum(s) / len(s)

    tw = [len((t["subject"] + " " + t["description"]).split()) for t in tickets]
    mn, md, mx, avg = wstats(tw)
    A(fmt_table(["Testo embeddato", "min", "mediana", "media", "max", "max token stimati"],
                [["ticket (subject+description)", mn, md, f"{avg:.0f}", mx, f"~{mx * TOKENS_PER_WORD:.0f}"]]) + "\n")

    sw = [len(s["text"].split()) for s in sections]
    mn2, md2, mx2, avg2 = wstats(sw)
    A(f"**Policy**: {len(sections)} sezioni da {len(set(s['file'] for s in sections))} file.\n")
    A(fmt_table(["Testo embeddato", "min", "mediana", "media", "max", "max token stimati"],
                [["sezione di policy", mn2, md2, f"{avg2:.0f}", mx2, f"~{mx2 * TOKENS_PER_WORD:.0f}"]]) + "\n")
    A("> Entrambi i tipi di chunk restano ampiamente sotto il limite di **2048 "
      "token** per input di `gemini-embedding-001`: non serve alcuna strategia "
      "di troncamento o di sub-chunking.\n")

    # ------------------------------------------------------------- KB 2
    A("## 4. KB policy\n")
    rows = []
    for f in sorted(set(s["file"] for s in sections)):
        ss = [s for s in sections if s["file"] == f]
        rows.append([f"`{f}`", ss[0]["doc_title"].split(":")[0], len(ss),
                     "sì" if any("scalation" in s["section_title"] for s in ss) else "no"])
    A(fmt_table(["File", "Policy", "Sezioni", "Ha sez. Escalation Criteria"], rows) + "\n")
    A("Tutte le policy espongono una sezione esplicita di criteri di escalation: "
      "sono le regole che la logica di decisione dovrà implementare, con "
      "**POL-006 come policy master** (dichiara esplicitamente di prevalere in "
      "caso di conflitto con le altre).\n")

    # ------------------------------------------- indicazioni per l'eval set
    A("## 5. Indicazioni per i dataset di evaluation\n")
    A("Dall'analisi emergono tre vincoli di progettazione:\n")
    A("1. **Leakage.** Gli stessi ticket sono sia nella collection `kb_tickets` "
      "sia candidati come query di test: senza accorgimenti il retrieval "
      "recupererebbe il ticket identico a sé stesso. Va usato un approccio "
      "*leave-one-out*, escludendo a query time il `ticket_id` di origine.\n")
    A(f"2. **Numerosità.** Con {n_esc} soli casi positivi, uno split train/test "
      "classico lascerebbe una manciata di positivi nel test set. Meglio valutare "
      "su tutti i ticket in leave-one-out, riportando le metriche anche "
      "stratificate per categoria.\n")
    A("3. **Copertura.** Serve un secondo dataset, scritto a mano, per i casi "
      "che i dati storici non coprono: richieste ambigue, fuori scope (POL-008 "
      "→ HR/Legal), o senza alcun riscontro nelle due KB — cioè esattamente i "
      "casi in cui deve scattare POL-006 §4.\n")
    A("Metrica primaria suggerita per l'escalation: **recall sulla classe "
      "'escalate'**. Il costo degli errori è asimmetrico — non escalare un "
      "ticket che andava escalato (falso negativo) è molto più grave "
      "dell'escalare un ticket che l'agente avrebbe potuto risolvere.\n")

    report = "\n".join(L)
    (OUT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
    print(f"Report scritto in {OUT_DIR / 'eda_report.md'}")
    print(f"Grafici in {FIG_DIR}/")
    print(f"\nTicket analizzati: {n} | escalati: {n_esc} ({n_esc/n:.1%})")
    print(f"Sezioni di policy: {len(sections)}")


if __name__ == "__main__":
    main()
