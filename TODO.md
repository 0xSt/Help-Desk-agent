# TODO — step aperti

Stato dei lavori. Da aggiornare a ogni step completato: le voci fatte si
spostano in fondo, in "Completato", con una riga di sintesi.

Legenda priorità: **P0** blocca il resto · **P1** step principale · **P2** rifinitura.

---

## Sicurezza — da fare subito

- [ ] **Revocare la chiave API pubblicata su GitHub.** Il file `.env.example`
      committato nel repository pubblico conteneva una chiave Gemini vera
      (`AIzaSyD0Yo...`). Rimuoverla dal file non basta: resta nella storia di
      Git ed è già stata indicizzata. **L'unica azione efficace è revocarla e
      generarne una nuova** da Google AI Studio.
      Corretto nel codice: `.env.example` non contiene più chiavi e il compose
      torna a leggere `.env` (non versionato) invece di `.env.example`.

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

**Già emersi e corretti al primo avvio reale (2026-08-20):**
- [x] MLflow rifiutava con 403 le chiamate del backend (`Rejected request with
      invalid Host header: mlflow:5000`). MLflow 3.x valida l'header Host
      contro il DNS rebinding e accetta di default solo localhost e IP privati;
      il nome di servizio Docker non è nella lista. Risolto aggiungendo
      `--allowed-hosts` al comando del server.
- [x] La chiave Gemini non arrivava ai container nonostante fosse nel `.env`.
      Causa: in Compose `environment:` **sovrascrive** `env_file:`, quindi
      `GEMINI_API_KEY: ${GEMINI_API_KEY:-}` scriveva una stringa vuota ogni
      volta che l'interpolazione non trovava il valore. Risolto passando a
      `env_file:` e lasciando in `environment:` solo ciò che il deploy deve
      imporre. Aggiunta anche `config.describe_credentials()`, loggata
      all'avvio di backend e ingestion, perché l'assenza di chiave non
      produce un errore ma un fallback silenzioso.

Punti ancora da verificare:
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

- [ ] Eseguire `docker compose run --rm ingestion python -m app.diagnose` e
      risolvere il primo controllo che fallisce.
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
**Strumento pronto**: `python -m evaluation.calibrate_thresholds`. Confronta la
distribuzione dei punteggi delle query in dominio (ticket in leave-one-out) con
quelle fuori dominio e propone la soglia che le separa meglio. Non usa i casi
etichettati, così restano intatti come test set.

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

### 4. Harness di evaluation — **completato**
Tutte e quattro le suite sono implementate ed eseguibili
(`retrieval`, `escalation-a`, `escalation-b`, `answers`). Vedi la sezione
"Evaluation" del README per i comandi.

Resta aperto:
- [ ] **Eseguire l'evaluation vera con Gemini attivo.** Tutti i numeri
      raccolti finora vengono dalla modalità mock e non dicono nulla sul
      sistema reale.
- [ ] **Rimisurare dopo la modifica al prompt "risolvi in un messaggio".**
      Chiedere risposte autosufficienti cambia il testo generato e può
      spostare la confidenza dichiarata, quindi tocca proprio i segnali su cui
      si regge l'escalation §4. Le suite vanno rieseguite prima e dopo la
      modifica per misurarne l'effetto invece di supporlo. Il giudizio di
      groundedness è l'indicatore da sorvegliare: il rischio specifico di
      questo prompt è che il modello colmi i vuoti inventando procedura pur di
      non fare domande.
- [ ] **Confrontare un giudice diverso dal modello valutato.** `JUDGE_MODEL`
      è configurabile apposta: un giudice che condivide con l'esaminato gli
      stessi punti ciechi tende a non vederne gli errori.

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

## P2 — Documentazione per l'esame

- [x] **Diagramma di sequenza del flusso HITL** — `docs/diagrams/hitl_sequence.puml`,
      sorgente PlantUML versionato più PNG e SVG renderizzati.
- [x] **Diagrammi dei casi d'uso** — due diagrammi separati per i due insiemi
      di attori: `use_cases_operativi.puml` (Richiedente, Operatore) e
      `use_cases_amministrazione.puml` (Manutentore).
- [ ] **Diagramma dei componenti** — i quattro servizi e le interfacce fra
      loro; è dove si vede il reverse proxy.
- [x] **Diagrammi delle classi** — due viste: `classi_astratto.puml`
      (sottosistemi e responsabilità, con la separazione fra i tre store) e
      `classi_dettaglio.puml` (attributi e firme delle strutture di dominio).
- [ ] **Diagramma di attività** — la logica di `decide_escalation`, con
      swimlane modello / motore di regole per rendere visibile il principio
      "il modello osserva, il codice decide".
- [ ] **Documento di specifica dei requisiti** (IEEE 830 / ISO 29148) con
      matrice di tracciabilità requisito → policy di origine → componente →
      test che lo verifica.

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

## Note dalle prime misurazioni

Numeri raccolti **in modalità mock**, quindi non rappresentativi del sistema
con Gemini: servono solo a validare gli strumenti.

- La calibrazione riporta una separazione del 72,5% tra query in dominio e
  fuori dominio, sotto la soglia di accettabilità: conferma che con il
  fallback nessuna soglia funziona bene. Da rimisurare con Gemini attivo.
- Sulla suite di retrieval, `recall@3 = 0,17` contro `hit_rate@3 = 0,77`
  sugli stessi identici dati: è l'artefatto del tetto previsto, e conferma
  la scelta di `hit_rate` e `mrr` come metriche primarie.
- 14 casi su 31 non recuperano la policy attesa. Da capire, dopo il passaggio
  a Gemini, quanto sia colpa del retrieval e quanto della mappa di rilevanza.
- Suite B (mock): recall 0,73 e precision 1,00 su 23 casi. Il breakdown per
  trigger mostra dove si perde: `POL-008 §5` (redirect fuori scope verso
  HR/Legal) ha recall **0,00** e `POL-006 §3.5` (richiesta esplicita di un
  umano) 0,33, mentre i trigger §3.1/§3.2/§3.3 sono a 1,00. Sono proprio i
  segnali che dipendono dalla comprensione del testo, cioè quelli che il mock
  non sa produrre: il primo confronto da fare con Gemini attivo.
- Suite A (mock, campione 20): recall 1,00 ma precision 0,36, cioè forte
  sovra-escalation. Atteso con la soglia di grounding non ancora tarata.

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
