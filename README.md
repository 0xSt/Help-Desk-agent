# Help Desk — prototipo HITL con LangGraph

Prototipo di sistema di help desk IT per un'azienda informatica fittizia. Un agente AI risponde alle richieste di supporto;
quando l'argomento è sensibile (sicurezza, accessi) o il modello dichiara
bassa confidenza, il ticket viene **escalato** a un operatore umano, che lo
vede in una console dedicata con tutto il contesto e può correggere/approvare
la risposta prima che venga inviata all'utente.

Stack: **FastAPI**  + **LangGraph**
(orchestrazione con interrupt/resume) + **HTML/JS vanilla** (due interfacce,
nessun framework/build step).


## Avvio con Docker Compose 

```
docker compose up --build
```
| Servizio | URL | Ruolo |
|---|---|---|
| **frontend** (nginx) | http://localhost:8080/ | interfaccia utente |
| | http://localhost:8080/agent | console operatore |
| **backend** (FastAPI) | http://localhost:8000/docs | API e grafo LangGraph |
| **qdrant** | http://localhost:6333/dashboard | database vettoriale |
| **mlflow** | http://localhost:5000 | trace delle esecuzioni ed evaluation |

Sequenza di avvio, garantita dalle condizioni in `depends_on`:

```
qdrant (healthy) ──▶ ingestion (completato) ──▶ backend (healthy) ──▶ frontend
mlflow (healthy) ──────────────────────────────▶
```

**Il job di ingestion.** Gira a ogni `docker compose up`, prima del backend:
crea le collection se non esistono, altrimenti le **aggiorna in modo
incrementale**. Ogni punto porta nel payload l'hash del testo embeddato e
della configurazione di embedding, quindi al riavvio si ri-embeddano solo i
chunk nuovi o modificati e si cancellano quelli spariti dalla sorgente. Cambiare modello o dimensione dell'embedding
invalida gli hash e forza la ricostruzione.

Il backend non parte finché l'ingestion non è uscita con successo
(`condition: service_completed_successfully`): senza quel vincolo le prime
richieste troverebbero collection vuote e ogni ticket verrebbe escalato per
"retrieval senza appigli".

Funziona subito senza chiave API, in modalità mock. 
Scenario per provare in mock:
1. Da `/`, scrivi "I clicked a phishing link and entered my password" → il
   ticket va in escalation, l'utente vede un messaggio di attesa.
2. Apri `/agent`: il ticket è in coda, con l'elenco dei trigger che l'hanno
   causato e la clausola di policy di ciascuno.
3. Modifica o approva la bozza, invia.
4. Torna sulla scheda utente: entro ~3 secondi (polling) compare la risposta,
   marcata "✓ verified by an agent".

## Gestione delle dipendenze: uv

`pyproject.toml` dichiara le dipendenze, `uv.lock` fissa le versioni esatte ed
**è versionato**: è ciò che rende identico l'ambiente tra la macchina di
sviluppo e i container. Le immagini usano `uv sync --frozen`, che fallisce se
il lock non è allineato al `pyproject` invece di risolvere silenziosamente
versioni diverse da quelle testate.


## Provider: Google Gemini

Sia la generazione sia l'embedding passano da Gemini (SDK `google-genai`).

- **Generazione** — `gemini-3.1-flash` (configurabile).
- **Embedding** — `gemini-embedding-001`.

**Senza `GEMINI_API_KEY`** il sistema resta interamente eseguibile: risposte
mock ed embedding hash-based deterministico. 


## Tracing con MLflow

`app/tracing.py`. All'avvio del backend `mlflow.langchain.autolog()` strumenta
l'esecuzione del grafo: ogni chiamata a `/api/chat` produce **una trace** con
uno span per nodo. Sopra l'autolog ci sono span espliciti su
`retrieval.kb_docs`, `retrieval.kb_tickets` ed `escalation.decide`, cioè
esattamente i punti i cui input/output servono all'evaluation.

**MLflow è pensato come servizio a sé**: l'URI arriva da
`MLFLOW_TRACKING_URI` e in Docker Compose punterà al container dedicato. Se la
variabile non è impostata, MLflow scrive in locale su `./mlruns`.

> L'integrazione di MLFlow non copre ancora nativamente gli interrupt di
> LangGraph.

## Retrieval: `app/retrieval.py`

Le due knowledge base vivono in `app/knowledge_base/`:

- `policies/POL-001..008-*.md` — le policy IT, indicizzate in Qdrant come
  collection `kb_docs`, **una sezione (`## Titolo`) per punto**: ogni chunk
  porta con sé il titolo del documento come contesto, così resta
  comprensibile anche isolato dal resto del file.
- `past_tickets.json` — lo storico ticket (135 record), indicizzato come
  collection `kb_tickets`, **un ticket per punto**. Viene embeddato solo il
  "lato problema" (oggetto + descrizione, ciò a cui una nuova richiesta
  somiglierà), mentre risoluzione/categoria/priorità/esito di escalation
  finiscono nel payload da mostrare come contesto una volta recuperato il punto.

## I dataset

Tutto il materiale è **simulato**: policy, ticket e casi di prova sono stati
costruiti per il progetto e non rappresentano un'organizzazione reale.

### Knowledge base — `app/knowledge_base/`

È ciò che il sistema **indicizza e interroga** per rispondere.

| File | Contenuto | Indicizzazione |
|---|---|---|
| `policies/POL-001..008-*.md` | 8 policy IT che definiscono procedure, soglie di approvazione e criteri di escalation | 60 chunk in `kb_docs`, uno per sezione `##` |
| `past_tickets.json` | 135 ticket risolti, 18 campi ciascuno | 135 punti in `kb_tickets`, uno per ticket |

Le policy sono la fonte normativa: ogni trigger di escalation cita la clausola
da cui deriva. POL-006 è la policy principale e prevale in caso di conflitto.

I ticket coprono 7 categorie e 20 sottocategorie, con distribuzione
volutamente sbilanciata come in un help desk reale (da 33 casi di gestione
accessi a 7 di collaborazione cloud).

#### Modello dati di un ticket

| Campo | Tipo | Ruolo |
|---|---|---|
| `ticket_id` | `str` | identificativo, es. `TCK-2026-00323`; diventa `source` nel payload Qdrant |
| `created_at` | `str` (ISO) | data di apertura |
| `source_channel` | `str` | canale: Chat, Email, Phone, Self-Service Portal |
| `department` | `str` | reparto del richiedente |
| `requester_role` | `str` | Standard Employee, Manager, Director |
| `category` | `str` | una delle 7 categorie |
| `subcategory` | `str` | una delle 20 sottocategorie |
| `priority` | `str` | da `P1` a `P4` |
| `subject` | `str` | oggetto della richiesta — **embeddato** |
| `description` | `str` | problema come descritto dall'utente — **embeddato** |
| `resolution_steps` | `list[str]` | passi seguiti dall'operatore |
| `resolution_summary` | `str` | sintesi della risoluzione |
| `resolution_time_minutes` | `int` | tempo impiegato |
| `status` | `str` | Resolved, Escalated_Resolved, Escalated_Pending |
| `was_escalated_to_human` | `bool` | **etichetta di riferimento** per la valutazione |
| `escalation_reason` | `str \| null` | motivazione, valorizzata solo se escalato |
| `csat_score` | `int \| null` | gradimento 1–5, assente sui ticket più critici |
| `tags` | `list[str]` | parole chiave |

#### Modello dati di una policy

Documenti Markdown con intestazione (`Version`, `Effective Date`, `Owner`,
`Applies To`) e sezioni numerate `## `. Ogni sezione diventa un chunk con
payload `{policy_id, policy_title, section_title, text, source}`, dove `text`
porta in testa il titolo del documento: un chunk recuperato isolatamente resta
così comprensibile.

Di ogni ticket viene embeddato il solo **lato problema** — oggetto e
descrizione, ciò a cui una nuova richiesta somiglia — mentre risoluzione,
categoria ed esito restano nel payload, come contesto da mostrare al modello
una volta recuperato il punto.

### Dati di valutazione — `eval_suite/data/`

Non vengono mai indicizzati: servono a **misurare** metriche.

| File | Contenuto | Usato da |
|---|---|---|
| `escalation_cases.json` | 43 casi scritti a mano con esito atteso e clausole attese: 23 da escalare, 20 da risolvere | suite `escalation` |
| `policy_relevance.json` | mappa delle 20 sottocategorie verso le policy attese e quelle accettabili | misure di pertinenza del recupero |

#### Modello dati di un caso di valutazione

| Campo | Tipo | Ruolo |
|---|---|---|
| `case_id` | `str` | identificativo, es. `ESC-015` |
| `query` | `str` | la richiesta sottoposta al sistema |
| `expected_escalate` | `bool` | esito atteso: è la verità di riferimento |
| `expected_trigger_codes` | `list[str]` | clausole che dovrebbero scattare, es. `["POL-006 §3.1"]` |
| `trigger_family` | `str` | `mandatory`, `confidence`, `retrieval` o `none` |
| `threshold_sensitive` | `bool` | se l'esito dipende dalle soglie configurate |
| `rationale` | `str` | perché quell'esito è corretto — documentazione, non usata dal codice |

`expected_trigger_codes` è ciò che permette di misurare non solo *se* la
decisione è corretta ma *per quale motivo*, distinguendo una decisione giusta
presa per la ragione sbagliata. `trigger_family` separa i segnali che
dipendono dal prompt da quelli che dipendono dalle soglie.

`policy_relevance.json` mappa ogni sottocategoria a
`{"expected": list[str], "acceptable": list[str]}`, dove le policy
`acceptable` sono pertinenti ma non indispensabili e non vengono conteggiate
come errore.

## Pipeline di evaluation: `eval_suite/`

Due suite separate, perché misurano cose che si guastano in modo indipendente:
una media unica nasconderebbe quale delle due non funziona.

### Parametri tracciati del sistema ad ogni run (valori di default)
| Parametro | Valore |
|---|---|
| prompt/agent_version |    1 |
| prompt/agent_uri |        prompts:/helpdesk-agent-system/1 |
| eval/judge_model |        gemini:/gemini-3.1-flash-lite |
| eval/dataset |            escalation_cases |
| llm_model |               gemini-3.1-flash-lite |
| confidence_threshold |    0.65 |

```bash
python -m eval_suite.run --suite escalation      # decisione
python -m eval_suite.run --suite quality --sample 20   # contesto e risposta
python -m eval_suite.run --suite all --sample 20
```

**`escalation`** — la decisione di coinvolgere un operatore è corretta? Valutata
sui 43 casi etichettati in `eval_suite/data/escalation_cases.json` con regole
deterministiche, **senza giudice**: l'esito atteso è annotato, quindi
introdurre un modello per stabilirlo aggiungerebbe solo rumore e costo.

Classe positiva: «va escalato». La metrica primaria è il **richiamo**, non
l'accuratezza: con circa un quarto di positivi, un sistema che non escala mai
raggiungerebbe il 76% pur essendo inutile. La gerarchia riflette l'asimmetria
dei costi — un falso negativo è un incidente di sicurezza mai rivisto, un falso
positivo qualche minuto di lavoro di un operatore.

| Metrica | Ruolo |
|---|---|
| `recall` | primaria: quota di ticket da escalare effettivamente colti |
| `precision` | costo operativo: quante escalation erano superflue |
| `f2` | sintesi, pesa il richiamo quattro volte la precisione |

**`quality`** — il contesto recuperato è pertinente e la risposta vi si attiene?
Valutata sui ticket storici in leave-one-out, con tre **scorer nativi di
MLflow**.

| Scorer | Cosa misura |
|---|---|
| `RetrievalRelevance` | i documenti recuperati sono pertinenti alla richiesta |
| `RetrievalGroundedness` | la risposta è sostenuta dal contesto recuperato |
| `RelevanceToQuery` | la risposta affronta la domanda posta |

I primi due si leggono insieme: fondatezza bassa con pertinenza alta indica che
il modello inventa pur avendo il materiale giusto; entrambe basse indicano che
il problema è a monte, nel retrieval.

### Esecuzione nei container

La suite gira dentro l'immagine del backend, che la contiene già e che ha
`MLFLOW_TRACKING_URI=http://mlflow:5000`: i risultati finiscono quindi
nell'istanza MLflow del compose e nei suoi volumi.

```bash
# 1. servizi necessari: Qdrant per il retrieval, MLflow per i risultati
docker compose up -d qdrant mlflow ingestion

# 2. valutazione
docker compose run --rm backend python -m eval_suite.run --suite escalation
docker compose run --rm backend python -m eval_suite.run --suite quality --sample 20
docker compose run --rm backend python -m eval_suite.run --suite all --sample 20

# 3. risultati
#    http://localhost:5000 -> esperimento "helpdesk-agent-eval"
```

I risultati persistono nei volumi `mlflow_db` (esperimenti, run, metriche,
tracce) e `mlflow_artifacts`.

## Riferimento API

| Endpoint | Metodo | Corpo | Usato da |
|---|---|---|---|
| `/api/chat` | POST | `{thread_id?, message}` | Utente |
| `/api/review` | POST | `{thread_id, edited_answer}` | Operatore |
| `/api/state/{thread_id}` | GET | — | Entrambe (polling utente, contesto ticket operatore) |
| `/api/tickets` | GET | — | Operatore (coda escalation) |
| `/api/threads` | GET | — | Entrambe (selettore utente, lista "conversazioni attive" operatore) |
| `/api/threads/{id}/close` | POST | — | Operatore (chiude una conversazione) |

### Monitoraggio dei prompt

`app/prompts.py` registra i prompt nel **Prompt Registry di MLflow**:

- il prompt dell'agente viene registrato all'avvio del backend;
- quello del giudice all'avvio di una valutazione — una sua modifica sposta i
  punteggi senza che il sistema valutato sia cambiato, quindi va tracciata
  anch'essa;
- l'alias `production` punta sempre alla versione attiva;
- la versione finisce fra i parametri di ogni run, di avvio e di valutazione.

