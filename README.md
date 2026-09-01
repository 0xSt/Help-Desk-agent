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

- **Generazione** — `gemini-3.1-flash` (configurabile). Usa lo **structured
  output nativo**: si passa uno schema Pydantic in `response_schema` e la
  risposta è garantita conforme, invece di chiedere "rispondi solo JSON" e
  sperare che il parsing regga. I parametri di sampling (`temperature`,
  `top_p`, `top_k`) non vengono impostati: sono deprecati sui modelli 3.x.
- **Embedding** — `gemini-embedding-001` con **task type asimmetrici**:
  `RETRIEVAL_DOCUMENT` in indicizzazione, `RETRIEVAL_QUERY` in query.

**Senza `GEMINI_API_KEY`** il sistema resta interamente eseguibile: risposte
mock ed embedding hash-based deterministico. Serve a sviluppare e testare
senza credenziali né costi, ma la qualità del retrieval non è rappresentativa.


## Tracing con MLflow

`app/tracing.py`. All'avvio del backend `mlflow.langchain.autolog()` strumenta
l'esecuzione del grafo: ogni chiamata a `/api/chat` produce **una trace** con
uno span per nodo. Sopra l'autolog ci sono span espliciti su
`retrieval.kb_docs`, `retrieval.kb_tickets` ed `escalation.decide`, cioè
esattamente i punti i cui input/output servono all'evaluation.

**MLflow è pensato come servizio a sé**: l'URI arriva da
`MLFLOW_TRACKING_URI` e in Docker Compose punterà al container dedicato. Se la
variabile non è impostata, MLflow scrive in locale su `./mlruns`. Il tracing
**non può far cadere una richiesta**: se il server è irraggiungibile, il
sistema continua a rispondere senza tracciare.

> Limite noto: quando il grafo si sospende su `interrupt()`, il tracer di
> MLflow emette un warning (`MlflowLangchainTracer has no attribute
> 'on_interrupt'`). È innocuo — la trace viene registrata comunque — ma
> segnala che l'integrazione non copre ancora nativamente gli interrupt di
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
- una **nuova versione nasce solo se il testo è effettivamente cambiato**:
  senza questo confronto ogni riavvio ne creerebbe una identica alla
  precedente, rendendo illeggibile la cronologia proprio quando serve;
- l'alias `production` punta sempre alla versione attiva;
- la versione finisce fra i parametri di ogni run, di avvio e di valutazione.

### Scelte di progetto

**Leave-one-out ovunque si usino i ticket come query.** Non solo perché una
query recupererebbe sé stessa, ma per una ragione più seria: il payload dei
ticket contiene `Escalated to a human agent: yes`, quindi senza esclusione il
modello leggerebbe nel contesto la risposta esatta alla domanda che gli stiamo
ponendo. Misureremmo la capacità di copiare, non di decidere.
