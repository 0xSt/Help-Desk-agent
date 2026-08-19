# Help Desk — prototipo HITL con LangGraph

Prototipo di sistema di help desk IT per un'azienda informatica fittizia
(volutamente senza nome). Un agente AI risponde alle richieste di supporto;
quando l'argomento è sensibile (sicurezza, accessi) o il modello dichiara
bassa confidenza, il ticket viene **escalato** a un operatore umano, che lo
vede in una console dedicata con tutto il contesto e può correggere/approvare
la risposta prima che venga inviata all'utente.

**Lingua del sistema: inglese.** Prompt del modello, interfacce utente e
operatore, messaggi generati dal backend sono tutti in inglese, coerentemente
con la knowledge base (policy + storico ticket) anch'essa in inglese. I
commenti nel codice e questo README restano in italiano: sono
documentazione per chi sviluppa il progetto, non output del sistema stesso.

Stack: **FastAPI** (backend, un solo processo per ora) + **LangGraph**
(orchestrazione con interrupt/resume) + **HTML/JS vanilla** (due interfacce,
nessun framework/build step).

> Progetto iterativo: questo README descrive lo stato attuale. Il retrieval
> sulle due knowledge base è **reale** (Qdrant locale, indice costruito dai
> file in `app/knowledge_base/`). La logica di decisione dell'escalation è
> ancora quella "semplice" del prototipo iniziale — vedi la sezione
> "Roadmap" in fondo.

## Avvio con Docker Compose (modalità consigliata)

```bash
cp .env.example .env        # e valorizza GEMINI_API_KEY
docker compose up --build
```

| Servizio | URL | Ruolo |
|---|---|---|
| **frontend** (nginx) | http://localhost:8080/ | interfaccia utente |
| | http://localhost:8080/agent | console operatore |
| **backend** (FastAPI) | http://localhost:8000/docs | API e grafo LangGraph |
| **qdrant** | http://localhost:6333/dashboard | database vettoriale |
| **mlflow** | http://localhost:5000 | trace delle esecuzioni |

Sequenza di avvio, garantita dalle condizioni in `depends_on`:

```
qdrant (healthy) ──▶ ingestion (completato) ──▶ backend (healthy) ──▶ frontend
mlflow (healthy) ──────────────────────────────▶
```

**Il job di ingestion.** Gira a ogni `docker compose up`, prima del backend:
crea le collection se non esistono, altrimenti le **aggiorna in modo
incrementale**. Ogni punto porta nel payload l'hash del testo embeddato e
della configurazione di embedding, quindi al riavvio si ri-embeddano solo i
chunk nuovi o modificati e si cancellano quelli spariti dalla sorgente. Senza
modifiche alla knowledge base il costo in chiamate all'API è **zero** (invece
di 168 embedding a ogni avvio). Cambiare modello o dimensione dell'embedding
invalida gli hash e forza la ricostruzione, che è il comportamento corretto:
vettori di provider diversi vivono in spazi incompatibili.

Il backend non parte finché l'ingestion non è uscita con successo
(`condition: service_completed_successfully`): senza quel vincolo le prime
richieste troverebbero collection vuote e ogni ticket verrebbe escalato per
"retrieval senza appigli".

**Perché il frontend è nginx con reverse proxy.** Separando il frontend in un
servizio a sé, le pagine sarebbero servite da un'origine diversa da quella
dell'API: CORS da configurare e URL del backend da iniettare nel JavaScript.
Con nginx che inoltra `/api/` al backend, browser e pagine vedono una sola
origine e **il codice JavaScript resta identico** a quello che gira in
sviluppo locale, dove è FastAPI a servire tutto.

## Avvio locale senza Docker

```bash
uv sync                     # oppure: pip install -r requirements.txt
uv run uvicorn app.main:app --reload
```

- `http://localhost:8000/` — interfaccia **utente**
- `http://localhost:8000/agent` — interfaccia **operatore**

In locale FastAPI serve anche le pagine statiche e `app/retrieval.py`
indicizza automaticamente all'import (`AUTO_INDEX=true` di default), quindi
non serve lanciare l'ingestion a mano. Qdrant gira in modalità *embedded*
(`QDRANT_PATH`, un indice su disco) e MLflow scrive su `./mlruns`: nessun
servizio esterno da avviare.

Funziona subito senza chiave API, in modalità mock. Prova questo scenario
end-to-end:
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

`matplotlib`, usata solo da `analysis/eda.py`, sta in un extra `analysis` e
non entra nell'immagine del backend:

```bash
uv sync --extra analysis && uv run python analysis/eda.py
```

## Architettura del grafo

```
           ┌──────────────────┐
    ┌─────▶│ retrieve_kb_docs  │────┐
    │      └──────────────────┘    │
  START                             ▼           escalate=False
    │      ┌─────────────────────┐ ┌───────┐   ┌──────────────────┐   ┌──────────┐
    └─────▶│ retrieve_kb_tickets │▶│ agent │──▶│ decide_escalation │──▶│ finalize │──▶ END
           └─────────────────────┘ └───────┘   └──────────────────┘   └──────────┘
                                                      │ escalate=True      ▲
                                                      ▼                    │
                                               ┌────────────────┐          │
                                               │ human_review    │─────────┘
                                               │ (interrupt())   │
                                               └────────────────┘
```

- **`retrieve_kb_docs` / `retrieve_kb_tickets`** — partono in parallelo da
  START (fan-out) e confluiscono su `agent` (fan-in): LangGraph aspetta che
  entrambi abbiano scritto la propria chiave di stato, dato che non c'è
  conflitto tra `kb_docs_context` e `kb_tickets_context`.
- **`agent`** — genera la bozza e **osserva** il ticket, restituendo segnali
  strutturati (categoria, priorità, approvazioni documentate, impatto
  multi-utente, richiesta esplicita di un umano...). Non decide l'escalation.
- **`decide_escalation`** — applica le regole di policy sui segnali. Vedi la
  sezione dedicata più sotto.
- **`human_review`** — nodo HITL: `interrupt()` sospende il grafo, il ticket
  entra nella coda dell'operatore con allegato l'elenco completo dei trigger.
- **`finalize`** — consolida la risposta definitiva e aggiorna la cronologia.

## Logica di escalation multisegnale

`app/escalation.py`. Principio di progetto: **il modello osserva, il codice
decide**. POL-006 §3 impone certe escalation *"regardless of the AI agent's
confidence level"*: una regola così non è implementabile se è il modello
stesso a decidere: dev'essere codice ispezionabile che nessuna risposta del
modello può aggirare.

Tre famiglie di segnali, in ordine di forza:

| Famiglia | Fonte | Esempi di trigger |
|---|---|---|
| **Mandatori** | POL-006 §3, POL-008 §5 | categoria Security; terminazione involontaria; accesso a sistema sensibile senza doppia approvazione; spesa oltre soglia; software fuori catalogo; richiesta esplicita di un umano; impatto multi-utente; richiesta fuori scope da redirigere a HR/Legal |
| **Confidenza** | POL-006 §4 | confidenza dichiarata sotto **0.65** (valore preso alla lettera dalla policy) |
| **Retrieval** | POL-006 §4 e §6 | *grounding*: nessun passaggio di policy **né** ticket storico sopra la similarità minima; *precedente*: tra i ticket storici molto simili, la maggioranza fu escalata da un umano |

La combinazione è un **OR su tutti i trigger**, deliberatamente conservativa:
il costo degli errori è asimmetrico, non escalare un ticket che andava
escalato è molto più grave dell'inverso. Ogni trigger porta con sé la clausola
che lo giustifica (`POL-006 §3.1`, ...), il che serve sia all'operatore (vede
*perché* gli è arrivato quel ticket) sia all'evaluation (permette di misurare
l'accuratezza per singolo trigger, non solo quella complessiva).

> **Soglie da ritarare.** `ESCALATION_MIN_RETRIEVAL_SCORE` è oggi calibrata
> sull'embedding di fallback e va rimisurata sui punteggi reali di Gemini: le
> scale di similarità di due provider diversi non sono confrontabili. Con
> l'embedding di fallback la soglia scatta quasi sempre, producendo
> sovra-escalation — è atteso, non un bug.

## Provider: Google Gemini

Sia la generazione sia l'embedding passano da Gemini (SDK `google-genai`).

- **Generazione** — `gemini-3.7-flash` (configurabile). Usa lo **structured
  output nativo**: si passa uno schema Pydantic in `response_schema` e la
  risposta è garantita conforme, invece di chiedere "rispondi solo JSON" e
  sperare che il parsing regga. I parametri di sampling (`temperature`,
  `top_p`, `top_k`) non vengono impostati: sono deprecati sui modelli 3.x.
- **Embedding** — `gemini-embedding-001` con **task type asimmetrici**:
  `RETRIEVAL_DOCUMENT` in indicizzazione, `RETRIEVAL_QUERY` in query. Non è un
  dettaglio: i due tipi producono rappresentazioni pensate per i due lati
  della stessa ricerca, usarne uno solo degrada il ranking. Dimensione ridotta
  a 768 via Matryoshka, con **ri-normalizzazione L2 esplicita** — un vettore
  unitario a 3072 dimensioni non resta unitario se lo tronchi, e le soglie di
  escalation si basano sul confronto tra punteggi.

**Senza `GEMINI_API_KEY`** il sistema resta interamente eseguibile: risposte
mock ed embedding hash-based deterministico. Serve a sviluppare e testare
senza credenziali né costi, ma la qualità del retrieval non è rappresentativa.

I due provider producono spazi vettoriali incompatibili. `ensure_index()` se
ne accorge confrontando la dimensione della collection esistente con quella
del provider attivo, e **ricostruisce l'indice da zero** invece di fallire con
un errore oscuro a query time.

## Tracing con MLflow

`app/tracing.py`. All'avvio del backend `mlflow.langchain.autolog()` strumenta
l'esecuzione del grafo: ogni chiamata a `/api/chat` produce **una trace** con
uno span per nodo. Sopra l'autolog ci sono span espliciti su
`retrieval.kb_docs`, `retrieval.kb_tickets` ed `escalation.decide`, cioè
esattamente i punti i cui input/output servono all'evaluation.

Gerarchia prodotta da un'invocazione:

```
LangGraph
├── retrieve_kb_docs      └── retrieval.kb_docs
├── retrieve_kb_tickets   └── retrieval.kb_tickets
├── agent
├── decide_escalation     └── escalation.decide
├── route_after_decision
└── human_review
```

La configurazione attiva (modelli, soglie, top-k) viene loggata come parametri
di un run `service-startup`, così è sempre ricostruibile *con quale
configurazione* è stato prodotto un certo insieme di trace.

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
- `past_tickets.json` — lo storico ticket (108 record), indicizzato come
  collection `kb_tickets`, **un ticket per punto**. Viene embeddato solo il
  "lato problema" (oggetto + descrizione, ciò a cui una nuova richiesta
  somiglierà), mentre risoluzione/categoria/priorità/esito di escalation
  finiscono nel payload da mostrare come contesto una volta recuperato il punto.

L'intero corpus è **materiale simulato** costruito per questo progetto
universitario: policy e ticket sono scenari fittizi, non esiste distinzione
tra dati "reali" e "sintetici".

## Analisi esplorativa

`analysis/eda.py` produce `analysis/eda_report.md` e i grafici in
`analysis/figures/`. Copre la categorizzazione dei ticket, la distribuzione
della ground truth di escalation, la mappatura di ogni escalation storica sul
trigger POL-006 §3 corrispondente, i vincoli di lunghezza per l'embedding e
le indicazioni per la costruzione dei dataset di evaluation. Si rigenera con:

```bash
python analysis/eda.py
```

Il corpus è stato ampliato da 54 a 108 ticket mantenendo **identiche** la
distribuzione per categoria e per sottocategoria e la proporzione di
escalation per categoria (26/108, 24,1%). Le etichette di escalation sono
tutte riconducibili a un trigger di POL-006 §3, quindi il corpus resta
utilizzabile come ground truth per la parte deterministica della logica.

**Qdrant in modalità locale**: `QdrantClient(path="qdrant_data")` non
richiede un server esterno — scrive l'indice su disco nel processo stesso.
È lo stesso client che in un secondo momento punterà a un servizio Qdrant
containerizzato nel docker-compose finale (cambia solo l'argomento del
costruttore, `path=...` diventa `url=...`).

**Indicizzazione automatica**: al primo import di `app/retrieval.py` (quindi
al primo avvio del processo), se le collection non esistono ancora vengono
create e popolate leggendo i file bundled nel progetto. Le esecuzioni
successive riusano l'indice già su disco — nessuna re-indicizzazione ad ogni
avvio. Cancella la cartella `qdrant_data/` per forzare una re-indicizzazione
(utile se aggiorni i file della knowledge base).

**Embedding.** Il provider attivo è Gemini (vedi la sezione "Provider" più
sopra). Senza chiave API si ricade su un hashing trick deterministico che
cattura sovrapposizione lessicale ma non significato: fa girare la pipeline a
costo zero, ma non è rappresentativo. Un esempio concreto della differenza:
con il fallback la query "my laptop battery drains very fast" restituisce come
primo risultato un ticket di *furto* di laptop, agganciandosi al token
"laptop" e ignorando il senso della richiesta. È un buon caso di regressione
da riusare come baseline per verificare il guadagno degli embedding reali.

## Le due interfacce e come condividono il backend

Un solo processo FastAPI serve due pagine statiche distinte, entrambe sempre
attive:

| Interfaccia | Route | File | Ruolo |
|---|---|---|---|
| Utente | `GET /` | `app/static/index.html` | Apre un ticket, aspetta (con polling) che venga risolto |
| Operatore | `GET /agent` | `app/static/agent.html` | Vede la coda dei ticket in escalation, modifica/approva la risposta |

Condividono lo stesso set di endpoint API (`/api/chat`, `/api/review`,
`/api/state/{id}`) più uno nuovo, `/api/tickets`, usato solo dalla console
operatore per popolare la coda.

**Perché l'utente non modifica più direttamente la bozza** (comportamento
del prototipo precedente): ora l'escalation va a un *altro* attore (un
operatore, non l'utente stesso). L'interfaccia utente, quando il ticket è in
attesa, fa **polling** su `GET /api/state/{thread_id}` ogni 3 secondi finché
lo stato non diventa `"completed"`. Non ho introdotto WebSocket/SSE per ora
— è la prossima ottimizzazione naturale se serve una notifica istantanea
invece che un ritardo di qualche secondo.

## `TicketStore`: perché uno store separato dal checkpointer di LangGraph

`app/store.py` introduce un `TicketStore` in memoria, deliberatamente
**separato** dal checkpointer LangGraph (`InMemorySaver`). Sono due concetti
diversi:

- il **checkpointer** sa "a che punto è l'esecuzione del grafo per il
  thread X" — serve per riprendere un `interrupt()`, non per essere
  interrogato con query tipo "dammi tutti i ticket aperti" (LangGraph non è
  pensato per quel tipo di accesso);
- il **`TicketStore`** rappresenta il record di business "ticket" come lo
  vedrebbe un vero help desk: oggetto, stato, motivo di escalation. In un
  sistema reale sarebbe una tabella su database.

`ticket_id` coincide sempre con `thread_id`: sono lo stesso identificatore
usato per due scopi — nessuna tabella di mapping serve. `main.py` aggiorna
il `TicketStore` negli stessi punti in cui già parla al grafo (`chat()`
quando scatta l'escalation, `review()` quando l'operatore risolve).

## Struttura dei file

```
app/
├── main.py       # endpoint FastAPI, 2 route HTML, orchestrazione grafo + TicketStore
├── config.py      # configurazione centralizzata: modelli, soglie, URI servizi
├── graph.py       # StateGraph LangGraph: retrieval + agent + decisione + HITL + finalize
├── escalation.py   # logica multisegnale di escalation (regole di policy)
├── tracing.py      # setup del tracing MLflow
├── ingest.py       # entrypoint del job di ingestion (python -m app.ingest)
├── retrieval.py    # indicizzazione e query sulle due collection Qdrant
├── llm.py         # chiamata a Gemini (o mock): bozza, confidenza e segnali sul ticket
├── store.py        # TicketStore in-memory (coda per l'interfaccia operatore)
├── threads.py       # ThreadRegistry in-memory (tutte le conversazioni, attive/chiuse)
├── schemas.py       # modelli Pydantic (richieste/risposte + coda ticket + thread)
├── knowledge_base/
│   ├── policies/      # POL-001..008-*.md
│   └── past_tickets.json  # 108 ticket (corpus simulato unico)
└── static/
    ├── shared.css    # design tokens comuni alle due interfacce
    ├── index.html     # interfaccia UTENTE
    └── agent.html      # interfaccia OPERATORE

docker/
├── backend.Dockerfile   # backend e job di ingestion (stessa immagine)
├── frontend.Dockerfile  # nginx con le due pagine statiche
├── mlflow.Dockerfile    # tracking server
└── nginx.conf           # routing e reverse proxy verso il backend

analysis/
├── eda.py               # analisi esplorativa rigenerabile
├── eda_report.md        # report prodotto
└── figures/             # grafici

docker-compose.yml   # orchestrazione dei 4 servizi + job di ingestion
pyproject.toml       # dipendenze (uv)
uv.lock              # versioni fissate, versionato
```

## Riferimento API

| Endpoint | Metodo | Corpo | Usato da |
|---|---|---|---|
| `/api/chat` | POST | `{thread_id?, message}` | Utente |
| `/api/review` | POST | `{thread_id, edited_answer}` | Operatore |
| `/api/state/{thread_id}` | GET | — | Entrambe (polling utente, contesto ticket operatore) |
| `/api/tickets` | GET | — | Operatore (coda escalation) |
| `/api/threads` | GET | — | Entrambe (selettore utente, lista "conversazioni attive" operatore) |
| `/api/threads/{id}/close` | POST | — | Operatore (chiude una conversazione) |

## Selezione conversazioni (utente) e chiusura thread (operatore)

`app/threads.py` introduce un `ThreadRegistry`, un registro di **tutte** le
conversazioni (non solo quelle mai passate dalla coda di escalation),
separato — per lo stesso motivo del `TicketStore` — dal checkpointer
LangGraph.

- **Lato utente**: un selettore in `index.html` (`GET /api/threads`) elenca
  le conversazioni attive, permettendo di riprenderne una senza dover
  ricordare o salvare l'URL con `?thread=...`.
- **Lato operatore**: una sezione "Conversazioni attive" in `agent.html`
  elenca lo stesso insieme, con un tasto "Chiudi" per ciascuna (`POST
  /api/threads/{id}/close`). Da quel momento `POST /api/chat` rifiuta nuovi
  messaggi su quel `thread_id` con un 400 esplicito.

**Limite noto**: non c'è nozione di identità utente (nessuna
autenticazione) — le liste sono globali, non filtrate per "proprietario".
Da rivedere quando si introdurrà l'autenticazione.

**Limite noto (edge case)**: chiudere un thread ancora fermo su un
`interrupt()` (escalation non ancora risolta) lo rimuove dalla coda ticket,
ma il grafo sottostante resta "sospeso per sempre" nel checkpointer — non
esiste un modo pulito per annullare un interrupt pendente in LangGraph. Con
`InMemorySaver` non è un problema pratico (si perde comunque al riavvio), ma
va rivisto se si introduce un checkpointer persistente.

## Roadmap (prossimi step)

1. **Dataset di evaluation** — due artefatti distinti: la ground truth di
   escalation ricavabile dai 108 ticket (che copre però solo i trigger
   deterministici §3), più un set di casi scritti a mano per i criteri §4,
   che lo storico non copre. Vanno tenuti separati dalla KB: sono query con
   decisione attesa, non ticket da indicizzare.
2. **Evaluation con MLflow** — tre target da misurare separatamente:
   qualità del retrieval (recall@k, MRR in leave-one-out), accuratezza della
   decisione di escalation (metrica primaria: **recall sulla classe
   "escalate"**, dato il costo asimmetrico degli errori) e qualità delle
   risposte finali.
3. **Taratura delle soglie** — una volta attivi gli embedding Gemini,
   rimisurare `ESCALATION_MIN_RETRIEVAL_SCORE` e le soglie del segnale
   "precedente" sui punteggi reali.
4. **Verifica del deploy su Docker** — i file di build e il compose sono
   scritti e validati staticamente, ma **non sono mai stati eseguiti**: vanno
   provati con un `docker compose up --build` reale (vedi "Limiti noti").

## Limiti noti

- **Il deploy Docker non è stato eseguito.** Dockerfile, `nginx.conf` e
  `docker-compose.yml` sono stati scritti e validati staticamente (sintassi
  YAML, grafo delle dipendenze, e `uv sync --frozen` provato davvero), ma
  nessun container è mai stato costruito o avviato. Le versioni delle immagini
  base e l'healthcheck di Qdrant — che usa il redirect bash su `/dev/tcp`,
  perché quell'immagine non include `curl` né `wget` — sono i punti più
  probabili da correggere al primo avvio reale.
- **Il percorso Gemini non è stato provato contro l'API vera**, per assenza di
  connettività verso `generativelanguage.googleapis.com` nell'ambiente di
  sviluppo. Il codice è scritto sulle firme reali dell'SDK `google-genai`, ma
  la prima esecuzione con una chiave valida è a tutti gli effetti un test da
  fare.
- **Le soglie di escalation non sono tarate.**
  `ESCALATION_MIN_RETRIEVAL_SCORE` è calibrata sull'embedding di fallback: con
  quello, scatta quasi sempre e produce sovra-escalation. Va rimisurata sui
  punteggi reali di Gemini.
- Quando il grafo si sospende su `interrupt()`, il tracer MLflow emette un
  warning (`MlflowLangchainTracer has no attribute 'on_interrupt'`): innocuo,
  la trace viene registrata comunque.
- Nessuna autenticazione: chiunque raggiunga `/agent` può risolvere ticket, e
  le liste di conversazioni sono globali.

## Estensioni minori già identificate

- **Persistenza reale**: sostituire `InMemorySaver` (grafo) e `TicketStore`
  (ticket) con storage persistente — vanno cambiati insieme, dato che
  condividono lo stesso ciclo di vita "si perde tutto al riavvio".
- **Notifica istantanea** invece del polling (WebSocket/SSE) sull'interfaccia utente.
- **Autenticazione dell'operatore**: oggi chiunque raggiunga `/agent` può
  risolvere ticket; in un sistema reale l'endpoint `/api/review` andrebbe
  protetto e si vorrebbe tracciare *quale* operatore ha risolto cosa.
