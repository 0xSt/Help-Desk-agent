# `eval_suite/` — valutazione del sistema

Suite di valutazione ricostruita da zero, basata sugli **scorer nativi di
MLflow** e sui dati già presenti nel progetto. Nessun dato viene rigenerato.

## Cosa misura

| Suite | Domanda | Come |
|---|---|---|
| `escalation` | la decisione di coinvolgere un operatore è corretta? | regole deterministiche sui 43 casi etichettati |
| `quality` | il contesto recuperato è pertinente e la risposta vi si attiene? | tre scorer MLflow, giudicati da un modello |

Sono separate perché rispondono a domande diverse e si guastano in modo
indipendente: una media unica nasconderebbe quale dei due aspetti non
funziona.

## Comandi

```bash
python -m eval_suite.run --suite escalation
python -m eval_suite.run --suite quality --sample 20
python -m eval_suite.run --suite all --sample 20
```

La suite `escalation` non richiede credenziali quando il sistema gira in
modalità mock; `quality` sì, perché le sue tre metriche sono giudizi di un
modello.

## File

```
eval_suite/
├── datasets.py   caricamento dei dataset esistenti
├── metrics.py    metriche di escalation (funzioni pure, testabili offline)
├── pipeline.py   esecuzione del sistema e registrazione del contesto nella traccia
└── run.py        entrypoint, scorer, registrazione su MLflow
```

## Le metriche di escalation

Classe positiva: «il ticket va escalato».

| Metrica | Ruolo |
|---|---|
| `recall` | **primaria** — quota di ticket da escalare effettivamente colti |
| `precision` | costo operativo — quanta parte delle escalation era superflua |
| `f2` | sintesi che pesa il richiamo quattro volte la precisione |
| `mcc` | sintesi robusta allo sbilanciamento delle classi |
| `specificity` | quota di ticket risolvibili lasciati all'agente |
| `tp/fp/fn/tn` | conteggi grezzi, per la diagnosi |
| `clausola/<POL>/recall` | richiamo per singola clausola di policy |
| `famiglia/<tipo>/recall` | richiamo per famiglia di segnale |

La gerarchia deriva dall'asimmetria dei costi: un falso negativo è un incidente
di sicurezza mai rivisto, un falso positivo è qualche minuto di un operatore.

**L'accuratezza è riportata ma non è primaria.** Con circa un quarto di
positivi, un sistema che non escala mai raggiunge il 76% pur essendo inutile —
verificabile con le funzioni di `metrics.py`, che su quel caso limite
restituiscono `accuracy 0,76` a fronte di `recall 0,00` e `mcc 0,00`.

Il richiamo per clausola cattura un caso che la matrice di confusione conta
come successo: la **decisione giusta presa per il motivo sbagliato**.

## Le metriche di qualità

| Scorer | Cosa misura | Su cosa opera |
|---|---|---|
| `RetrievalRelevance` | i documenti recuperati sono pertinenti alla richiesta | span `RETRIEVER` della traccia |
| `RetrievalGroundedness` | la risposta è sostenuta dal contesto recuperato | span `RETRIEVER` + risposta |
| `RelevanceToQuery` | la risposta affronta la domanda posta | `inputs` e `outputs` |

I primi due vanno letti insieme: fondatezza bassa con pertinenza alta indica
che il modello inventa pur avendo il materiale giusto; entrambe basse indicano
che il problema è a monte, nel recupero.

## Due vincoli tecnici da conoscere

**Gli scorer di recupero leggono la traccia, non gli argomenti.** Cercano uno
span di tipo `RETRIEVER` e ne interpretano l'output come elenco di documenti.
Il grafo dell'applicazione non produce span di quel tipo, e `pipeline.py`
registra quindi uno span dedicato dentro la sola traccia di valutazione: il
codice di produzione resta intatto e la strumentazione appartiene a chi misura.

Conseguenza da tenere presente: gli scorer si possono applicare alle tracce
prodotte da questa suite, **non** a quelle del traffico reale, che non
contengono lo span. Per abilitarlo anche in esercizio andrebbero modificate le
funzioni di recupero in `app/retrieval.py`.

**Il giudice va indicato esplicitamente.** In assenza di indicazioni MLflow
userebbe un modello OpenAI; qui si passa `gemini:/<modello>`, che MLflow
instrada tramite LiteLLM. Il modello è preso da `JUDGE_MODEL`, configurabile
separatamente da quello che genera le risposte: un giudice che condivide con il
valutato gli stessi punti ciechi tende a non vederne gli errori.

## Riproducibilità

Ogni esecuzione registra come parametri del run la configurazione attiva, la
**versione del prompt** dell'agente con il suo URI nel registry, il modello
giudice e la composizione del dataset:

```
prompt/agent_version    1
prompt/agent_uri        prompts:/helpdesk-agent-system/1
eval/judge_model        gemini:/gemini-3.1-flash-lite
eval/dataset            escalation_cases
eval/n_cases            43
llm_model               gemini-3.1-flash-lite
embedding_provider      gemini
min_retrieval_score     0.45
confidence_threshold    0.65
```

È ciò che rende confrontabili due valutazioni a distanza di tempo: senza i
parametri che le hanno prodotte, due serie di metriche non sono interpretabili
e una differenza non è attribuibile al prompt piuttosto che alle soglie.
