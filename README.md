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

## Avvio rapido

```bash
pip install -r requirements.txt
cd hitl-langgraph-app
uvicorn app.main:app --reload
```

- `http://localhost:8000/` — interfaccia **utente**: apre un ticket.
- `http://localhost:8000/agent` — interfaccia **operatore**: coda ticket in escalation.

Funziona subito senza chiave API (modalità mock). Per usare Claude davvero,
copia `.env.example` in `.env` e imposta `ANTHROPIC_API_KEY`.

Prova questo scenario end-to-end:
1. Da `/`, scrivi "I can't access the company VPN" → il ticket va in escalation, l'utente vede un messaggio di attesa (senza poter modificare nulla).
2. Apri `/agent` (in un'altra scheda): il ticket compare in coda.
3. Clicca sul ticket, modifica o approva la bozza, invia.
4. Torna sulla scheda utente: entro ~3 secondi (polling) compare la risposta, marcata "✓ verificata da un operatore".

## Architettura del grafo

```
               ┌──────────────────┐
        ┌─────▶│ retrieve_kb_docs  │────┐
        │      └──────────────────┘    │
    START                               ▼
        │      ┌─────────────────────┐ ┌────────┐   needs_review=False   ┌──────────┐
        └─────▶│ retrieve_kb_tickets │▶│ agent  │────────────────────────▶ finalize │──▶ END
               └─────────────────────┘ └────────┘                       └──────────┘
                                            │ needs_review=True                ▲
                                            ▼                                  │
                                     ┌────────────────┐                       │
                                     │ human_review    │───────────────────────┘
                                     │ (interrupt())   │
                                     └────────────────┘
```

- **`retrieve_kb_docs` / `retrieve_kb_tickets`** — partono in parallelo da
  START (fan-out) e confluiscono entrambi su `agent` (fan-in, LangGraph
  aspetta che entrambi abbiano scritto la loro chiave di stato prima di
  eseguire `agent`, dato che non c'è conflitto tra `kb_docs_context` e
  `kb_tickets_context`). Interrogano davvero le due collection Qdrant —
  vedi la sezione dedicata più sotto.
- **`agent`** — genera la bozza (passando il contesto recuperato dalle due
  KB a `generate_draft_answer`) e applica la regola di escalation attuale
  (parole chiave sensibili + soglia di confidenza — **verrà rivista**).
- **`human_review`** — nodo HITL: `interrupt()` sospende il grafo, il ticket
  viene registrato nel `TicketStore` e diventa visibile in coda all'operatore.
- **`finalize`** — consolida la risposta definitiva e aggiorna la cronologia.

## Retrieval: `app/retrieval.py`

Le due knowledge base vivono in `app/knowledge_base/`:

- `policies/POL-001..008-*.md` — le policy IT, indicizzate in Qdrant come
  collection `kb_docs`, **una sezione (`## Titolo`) per punto**: ogni chunk
  porta con sé il titolo del documento come contesto, così resta
  comprensibile anche isolato dal resto del file.
- `past_tickets.json` — lo storico ticket, indicizzato come collection
  `kb_tickets`, **un ticket per punto**. Viene embeddato solo il "lato
  problema" (oggetto + descrizione, ciò a cui una nuova richiesta
  somiglierà), mentre risoluzione/categoria/priorità/esito di escalation
  finiscono nel payload da mostrare come contesto una volta recuperato il punto.

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

**Embedding — limite noto**: per non introdurre dipendenze pesanti (`torch`,
un modello locale) né richiedere una chiave API aggiuntiva per funzionare
"out of the box" (stesso principio della modalità mock in `llm.py`),
l'embedding usato oggi è un **hashing trick deterministico** puro Python
(bag-of-words proiettato via hash in un vettore a 256 dimensioni). Cattura
soprattutto sovrapposizione lessicale letterale, non vera similarità
semantica. Ora che tutto il sistema (prompt, UI, knowledge base) è in
inglese, la qualità è nettamente migliore rispetto a quando c'era un
disallineamento di lingua tra query (italiane) e KB (inglese) — nei test gli
score sono più che raddoppiati sulle stesse query tradotte in inglese — ma
resta un limite di sinonimia: una query come "my colleague is harassing me"
non recupera bene POL-008 (che pure ha la sezione esatta su questo, redirect
a HR) perché "harassing" non condivide token con "harassment"/"misconduct"
nel testo della policy. Un embedding reale (es. Voyage AI, che Anthropic
raccomanda per l'uso con Claude) risolverebbe anche questo: il punto di
innesto è `embed_text()` in `retrieval.py` — sostituirla, aggiornare
`EMBEDDING_DIM` alla nuova dimensione del vettore, cancellare `qdrant_data/`
e far ripartire l'indicizzazione.

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
├── graph.py       # StateGraph LangGraph: retrieval + agent + human_review + finalize
├── retrieval.py    # indicizzazione e query sulle due collection Qdrant
├── llm.py         # chiamata a Claude (o mock), persona "help desk IT", regole escalation
├── store.py        # TicketStore in-memory (coda per l'interfaccia operatore)
├── threads.py       # ThreadRegistry in-memory (tutte le conversazioni, attive/chiuse)
├── schemas.py       # modelli Pydantic (richieste/risposte + coda ticket + thread)
├── knowledge_base/
│   ├── policies/      # POL-001..008-*.md
│   └── past_tickets.json
└── static/
    ├── shared.css    # design tokens comuni alle due interfacce
    ├── index.html     # interfaccia UTENTE
    └── agent.html      # interfaccia OPERATORE
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

## Roadmap (prossimi step, non ancora implementati)

Questi punti sono discussi in dettaglio nella conversazione, qui solo un
riepilogo di cosa manca:

1. **Embedding reale** — sostituire l'hashing trick in `retrieval.py` con
   un provider vero (es. Voyage AI) per una qualità di retrieval migliore,
   in particolare sulla sinonimia (vedi limite noto sopra).
2. **Logica di decisione dell'escalation** — sostituire la regola attuale
   (parole chiave + soglia di confidenza) con qualcosa di più robusto,
   informato anche dai risultati del retrieval (es. nessun passaggio/ticket
   sopra una soglia di similarità minima -> escalation forzata, come
   previsto dalla policy POL-006 §4).
3. **Deploy con Docker Compose** — servizi separati: `frontend`, `backend`,
   `qdrant` (a quel punto non più locale ma un servizio containerizzato
   vero, cambiando solo l'argomento del client da `path=` a `url=`),
   `mlflow` (tracing/evaluation).

## Estensioni minori già identificate

- **Persistenza reale**: sostituire `InMemorySaver` (grafo) e `TicketStore`
  (ticket) con storage persistente — vanno cambiati insieme, dato che
  condividono lo stesso ciclo di vita "si perde tutto al riavvio".
- **Notifica istantanea** invece del polling (WebSocket/SSE) sull'interfaccia utente.
- **Autenticazione dell'operatore**: oggi chiunque raggiunga `/agent` può
  risolvere ticket; in un sistema reale l'endpoint `/api/review` andrebbe
  protetto e si vorrebbe tracciare *quale* operatore ha risolto cosa.
