# TODO — step aperti

Stato dei lavori. Da aggiornare a ogni step completato: le voci fatte si
spostano in fondo, in "Completato", con una riga di sintesi.

Legenda priorità: **P0** blocca il resto · **P1** step principale · **P2** rifinitura.

---

## P0 — Verifiche che sbloccano tutto il resto

Nessuna delle attività P1 produce risultati sensati finché queste non sono
fatte. Vanno in quest'ordine.

### 1. Primo avvio reale di Docker Compose
Nessun container è mai stato costruito né avviato: Docker non era disponibile
nell'ambiente in cui il progetto è stato sviluppato. Validati staticamente
solo la sintassi YAML, il grafo delle dipendenze e `uv sync --frozen`.

```bash
cp .env.example .env      # e valorizza GEMINI_API_KEY
docker compose up --build
```

Punti che più probabilmente richiederanno una correzione:
- [ ] **healthcheck di Qdrant** — usa `bash -c 'exec 3<>/dev/tcp/localhost/6333'`
      perché quell'immagine non include `curl` né `wget`. Se manca anche bash
      resta `unhealthy` e blocca la catena. Rimedio: disabilitarlo e affidarsi
      al retry con backoff già presente in `app/ingest.py`.
- [ ] **versioni delle immagini base** — `qdrant/qdrant:v1.12.4`,
      `nginx:1.27-alpine`, `python:3.12-slim`.
- [ ] **accesso a `ghcr.io/astral-sh/uv:latest`** nella build stage.
- [ ] **reverse proxy nginx** — verificare che `/api/*` arrivi davvero al
      backend e che le due UI funzionino da `:8080`.

### 2. Prima esecuzione con una chiave Gemini vera
Il percorso Gemini non è mai stato eseguito contro l'API reale (nessuna
connettività verso `generativelanguage.googleapis.com` in sviluppo). Il codice
è scritto sulle firme reali dell'SDK, ma resta non verificato.

- [ ] Verificare il **nome del modello**: il default `GEMINI_MODEL` è una
      scelta plausibile ma non confermata. Elencare i modelli disponibili:
      ```python
      from google import genai
      for m in genai.Client(api_key="...").models.list():
          print(m.name, m.supported_actions)
      ```
      Se il nome è sbagliato ogni chiamata fallisce e `llm.py` restituisce il
      fallback a confidenza 0.0, che **escala tutti i ticket**: sintomo
      confondibile con un errore di logica.
- [ ] Confermare nei log `Provider di embedding: gemini` (non
      `hashing-fallback`).
- [ ] Verificare che l'indice si ricostruisca da solo al passaggio
      256 → 768 dimensioni.
- [ ] Controllare il costo della prima indicizzazione (195 embedding: 60
      chunk di policy + 135 ticket).

### 3. Taratura delle soglie di escalation
`ESCALATION_MIN_RETRIEVAL_SCORE` è oggi calibrata sull'embedding di fallback e
**non è trasferibile**: le scale di similarità di provider diversi non sono
confrontabili. Con il valore attuale il trigger di grounding scatta quasi
sempre e il sistema sovra-escala.

- [ ] Raccogliere la distribuzione dei punteggi reali su query rappresentative
      (`aggregate_retrieval` restituisce già media, mediana e decimo
      percentile del top score proprio per questo).
- [ ] Scegliere la soglia sotto i punteggi delle query legittime e sopra
      quelli fuori dominio.
- [ ] Rivedere anche `ESCALATION_PRECEDENT_SCORE_FLOOR`.

---

## P1 — Evaluation

### 4. Completare l'harness (`evaluation/run_evaluation.py`)
Lo scheletro, il caricamento dei dataset e il logging MLflow ci sono. I `TODO`
numerati nel file corrispondono a queste voci:

- [ ] **TODO(1)** — invocare davvero il sistema nella suite di escalation.
      Preferire l'esecuzione del grafo con lettura di `escalation_triggers`
      dallo stato (misura il sistema com'è in produzione) rispetto alla
      chiamata diretta a `escalation.decide()`.
- [ ] **TODO(2)** — decidere se aggiungere i 135 ticket storici come suite
      separata. Sono ground truth vera ma coprono solo i trigger §3, e vanno
      usati in leave-one-out. Da non mescolare con i casi scritti a mano.
- [ ] **TODO(3)** — leave-one-out nel retrieval: escludere il ticket stesso
      dai risultati (filtro `must_not` su `source`, oppure recuperare k+1 e
      scartare l'auto-match).
- [ ] **TODO(4)** — definire la ground truth di rilevanza: per `kb_tickets` si
      può usare la stessa `subcategory` come proxy automatico; per `kb_docs`
      serve una mappatura categoria → policy attese scritta a mano.
- [ ] **TODO(5)** — qualità delle risposte con `mlflow.genai.evaluate()` e
      scorer custom di policy compliance.

### 5. Completare i dataset di evaluation
`evaluation/datasets/escalation_cases.json` ha 23 casi (15 positivi, 8
negativi).

- [ ] Aggiungere i casi §4 **dipendenti dalle soglie**, che ha senso scrivere
      solo dopo il punto 3 (sono marcati `threshold_sensitive` nello schema).
- [ ] Ampliare i negativi: servono a misurare la precision, e con la logica
      attuale in OR conservativo sono la parte a rischio.
- [ ] Valutare una revisione incrociata delle etichette attese: oggi sono
      derivate dalle policy da una sola persona.

### 6. Sweep delle soglie
- [ ] Confrontare configurazioni diverse come run MLflow.
      `config.as_params()` è già scritto per questo: logga tutte le soglie
      insieme alle metriche, così i run sono confrontabili a posteriori.

---

## P2 — Rifiniture

- [ ] **Pubblicare su GitHub** — il repository locale è pronto con la storia
      dei commit; mancano `git remote add origin ...` e `git push -u origin main`.
- [ ] **Feedback implicito dagli operatori** — la differenza tra
      `draft_answer` e la risposta editata dall'operatore è un segnale di
      qualità a costo zero, loggabile a ogni risoluzione. Il design HITL lo
      rende disponibile senza alcuna annotazione manuale.
- [ ] **Persistenza reale** — `InMemorySaver` (grafo), `TicketStore` e
      `ThreadRegistry` sono in RAM e si perdono al riavvio. Vanno sostituiti
      insieme, condividendo lo stesso ciclo di vita.
- [ ] **Autenticazione** — chiunque raggiunga `/agent` può risolvere ticket;
      le liste di conversazioni sono globali, senza nozione di proprietario.
- [ ] **Notifica istantanea** — sostituire il polling ogni 3s con WebSocket o
      SSE sull'interfaccia utente.
- [ ] **Warning MLflow sugli interrupt** — quando il grafo si sospende,
      il tracer emette `MlflowLangchainTracer has no attribute 'on_interrupt'`.
      Innocuo (la trace viene registrata), ma da monitorare.
- [ ] **Annullare un interrupt pendente** — chiudere un thread fermo su
      `interrupt()` lo toglie dalla coda ma lascia il grafo sospeso per sempre
      nel checkpointer. Irrilevante con `InMemorySaver`, da risolvere se si
      introduce un checkpointer persistente.

---

## Completato

- **Prototipo HITL** — workflow LangGraph con `interrupt()`/`Command(resume=)`,
  due interfacce (utente e operatore) servite dallo stesso backend, coda
  ticket, selezione e chiusura conversazioni.
- **Retrieval su due knowledge base** — collection Qdrant `kb_docs` (60 chunk,
  una sezione di policy per punto) e `kb_tickets` (135 ticket, embeddato il
  solo lato problema).
- **EDA** (`analysis/eda.py`) — rigenerabile; ha mostrato che le etichette di
  escalation storiche sono interamente spiegate dai trigger deterministici
  POL-006 §3, e che nessun ticket copre i criteri §4.
- **Estensione della KB** — da 54 a 135 ticket, distribuzione per categoria e
  sottocategoria preservata, tasso di escalation 24,1% → 24,4%.
- **Correzione dati** — TCK-2025-00145 e 00147 (Security non escalati) resi
  coerenti con POL-005 §8.
- **Passaggio a Gemini** — generazione con structured output nativo, embedding
  con task type asimmetrici e troncatura Matryoshka a 768 con
  ri-normalizzazione L2. Fallback deterministico senza chiave API.
- **Escalation multisegnale** (`app/escalation.py`) — trigger mandatori §3,
  confidenza §4, e due segnali da retrieval; ogni trigger porta la clausola di
  policy che lo giustifica.
- **Tracing MLflow** — una trace per invocazione, uno span per nodo più span
  espliciti su retrieval ed escalation.
- **Deploy** — Docker Compose a 4 servizi più job di ingestion incrementale
  (hash del contenuto: zero embedding sui riavvii senza modifiche),
  packaging con uv.
- **Metriche di evaluation** (`evaluation/metrics.py`) — funzioni pure e
  testate per retrieval, escalation e breakdown per trigger e per famiglia.
