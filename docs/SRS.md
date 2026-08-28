---
title: "Specifica dei Requisiti Software"
subtitle: "Sistema di help desk IT con agente conversazionale e revisione umana"
author: "Progetto universitario"
date: "Versione 1.0"
lang: it
documentclass: article
fontsize: 11pt
geometry: margin=2.5cm
toc: true
toc-depth: 2
numbersections: true
colorlinks: true
linkcolor: black
urlcolor: blue
---

\newpage

# Introduzione

## Scopo del documento

Questo documento specifica i requisiti del sistema di help desk IT sviluppato
come progetto d'esame. È redatto seguendo la struttura suggerita dallo standard
IEEE 830 / ISO-IEC-IEEE 29148, adattata alla dimensione del progetto: il
capitolo sui requisiti funzionali è volutamente il più esteso, mentre i
requisiti non funzionali sono trattati in forma sintetica.

Ogni requisito è **identificato**, **verificabile** e **tracciabile** verso la
sua origine: nella maggior parte dei casi una clausola delle policy aziendali
simulate che il sistema deve applicare.

## Ambito del sistema

Il sistema riceve richieste di supporto IT da parte di utenti, genera una
risposta con un modello linguistico assistito da recupero di informazioni
(RAG) su due basi di conoscenza, e decide autonomamente se la risposta possa
essere consegnata direttamente oppure debba essere **revisionata da un
operatore umano** prima dell'invio.

Il dominio applicativo, le policy e lo storico dei ticket sono **materiale
simulato** costruito per il progetto: non rappresentano un'organizzazione
reale. La simulazione non riduce il realismo dei requisiti, che derivano dal
testo delle policy esattamente come deriverebbero da quelle di un'azienda vera.

## Definizioni e acronimi

| Termine | Significato |
|---|---|
| **Agente** | Il componente automatico che genera la risposta al ticket |
| **Operatore** | La persona che revisiona le risposte escalate |
| **Ticket** | Una richiesta di supporto aperta da un utente |
| **Thread** | Una conversazione, che può contenere più turni |
| **Escalation** | Il passaggio di un ticket dall'agente a un operatore umano |
| **Trigger** | Una condizione che, se soddisfatta, impone l'escalation |
| **RAG** | *Retrieval-Augmented Generation*: generazione assistita dal recupero di documenti pertinenti |
| **HITL** | *Human-In-The-Loop*: presenza di un intervento umano nel flusso automatico |
| **Chunk** | Porzione di documento indicizzata come unità autonoma di recupero |
| **Embedding** | Rappresentazione vettoriale di un testo, usata per la ricerca semantica |
| **Collection** | Insieme di vettori indicizzati nel database vettoriale |
| **Confidenza** | Auto-valutazione del modello sull'affidabilità della propria risposta, in scala [0, 1] |
| **Grounding** | Fondatezza della risposta sul contesto effettivamente recuperato |
| **Leave-one-out** | Modalità di valutazione in cui l'elemento usato come query è escluso dai risultati |

## Riferimenti

Le policy aziendali simulate costituiscono la fonte primaria dei requisiti
funzionali:

| Codice | Titolo |
|---|---|
| POL-001 | Password and Account Access Management Policy |
| POL-002 | Access Request and Provisioning Policy |
| POL-003 | Hardware Issuance, Replacement and Repair Policy |
| POL-004 | Software Installation and Licensing Policy |
| POL-005 | Security Incident Response Policy |
| POL-006 | Ticket Priority, SLA and Escalation Policy (policy *master*) |
| POL-007 | Remote Access and VPN Policy |
| POL-008 | Acceptable Use and Data Handling Policy |

POL-006 dichiara esplicitamente di prevalere in caso di conflitto con le altre:
questa gerarchia è riflessa nei requisiti.

\newpage

# Descrizione generale

## Contesto

Il sistema è un'applicazione web autonoma, eseguibile in locale tramite
container. Dipende da due servizi esterni: un modello linguistico per
generazione ed embedding, e un database vettoriale per il recupero.

La base di conoscenza è composta da **8 documenti di policy** e **135 ticket
storici risolti**, entrambi in lingua inglese, indicizzati in due collection
distinte.

## Classi di utenti

Il sistema riconosce tre classi di utenti, con obiettivi e permessi diversi.

**Richiedente** — dipendente o collaboratore che apre un ticket. Descrive il
proprio problema e riceve una risposta. **Non ha alcuna facoltà di modificare
la risposta proposta dall'agente**: può solo consultarne l'esito. Questa
limitazione è deliberata e distingue il sistema da una comune chat assistita.

**Operatore** — addetto all'help desk. Riceve i ticket escalati corredati del
contesto completo, corregge o approva la bozza prodotta dall'agente, e può
chiudere definitivamente una conversazione.

**Manutentore** — chi amministra il sistema: popola la base di conoscenza,
configura le soglie, esegue la valutazione e consulta le tracce di esecuzione.

## Funzionalità principali

1. Apertura e gestione conversazionale dei ticket di supporto
2. Recupero di contesto da due basi di conoscenza distinte
3. Generazione di una risposta con dichiarazione di confidenza
4. Decisione automatica di escalation basata su segnali multipli
5. Console dedicata all'operatore per la revisione delle risposte
6. Popolamento e aggiornamento incrementale dell'indice vettoriale
7. Tracciamento delle esecuzioni e valutazione delle prestazioni

## Vincoli generali

- **Assenza di autenticazione**: il prototipo non implementa identità utente.
  Le liste di conversazioni e ticket sono globali.
- **Stato non persistente**: lo stato applicativo risiede in memoria di
  processo e si perde al riavvio.
- **Dipendenza da un servizio esterno**: la mancanza di connettività verso il
  provider del modello linguistico degrada, ma non interrompe, il servizio.
- **Lingua**: tutti i contenuti rivolti all'utente e i prompt del modello sono
  in lingua inglese, coerentemente con la base di conoscenza.

## Assunzioni e dipendenze

Si assume la disponibilità di un ambiente container, di credenziali valide per
il provider del modello linguistico, e che la base di conoscenza sia
staticamente definita da file versionati insieme al codice.

\newpage

# Requisiti funzionali

## Convenzioni

Ogni requisito è espresso nella forma:

> **RF-nn** — *Enunciato*. **Fonte**: origine del requisito. **Verifica**:
> modalità con cui se ne accerta il soddisfacimento.

Il verbo **DEVE** indica un requisito obbligatorio; **DOVREBBE** un requisito
raccomandato ma non vincolante. La colonna *Fonte* riporta la clausola di
policy da cui il requisito discende, oppure l'indicazione *progetto* quando si
tratta di una scelta architetturale non imposta dal dominio.

## Gestione delle conversazioni e dei ticket

**RF-01** — Il sistema DEVE consentire a un richiedente di aprire un ticket
descrivendo il proprio problema in linguaggio naturale, senza compilare campi
strutturati.
**Fonte**: progetto. **Verifica**: prova funzionale sull'interfaccia utente.

**RF-02** — Il sistema DEVE assegnare a ogni nuova conversazione un
identificativo univoco, generato automaticamente e restituito al client.
**Fonte**: progetto. **Verifica**: ispezione della risposta di `POST /api/chat`.

**RF-03** — Il sistema DEVE mantenere la cronologia dei turni di una
conversazione e renderla disponibile al modello nei turni successivi, senza che
il client debba ritrasmetterla.
**Fonte**: progetto. **Verifica**: due messaggi consecutivi sullo stesso
identificativo; il secondo deve poter fare riferimento al primo.

**RF-04** — Il sistema DEVE consentire di riprendere una conversazione
esistente a partire dal suo identificativo, ricostruendone la cronologia
completa.
**Fonte**: progetto. **Verifica**: `GET /api/state/{id}` restituisce la
cronologia; ricaricamento della pagina con conversazione in corso.

**RF-05** — Il sistema DEVE esporre l'elenco delle conversazioni attive.
**Fonte**: progetto. **Verifica**: `GET /api/threads`.

**RF-06** — Il sistema DEVE consentire a un operatore di chiudere
definitivamente una conversazione, indipendentemente dal fatto che sia mai
stata escalata.
**Fonte**: progetto. **Verifica**: `POST /api/threads/{id}/close`.

**RF-07** — Il sistema DEVE rifiutare, con esito esplicito, ogni nuovo
messaggio inviato su una conversazione chiusa.
**Fonte**: progetto. **Verifica**: invio di un messaggio dopo la chiusura; il
sistema risponde con errore 400 e messaggio comprensibile.

**RF-08** — Il sistema DEVE distinguere una conversazione mai esistita da una
conclusa, restituendo esiti differenti.
**Fonte**: progetto. **Verifica**: `GET /api/state/{id}` su identificativo
inesistente restituisce 404.

## Recupero del contesto

**RF-09** — Il sistema DEVE indicizzare i documenti di policy in una
collection dedicata, suddividendoli in chunk corrispondenti alle sezioni del
documento.
**Fonte**: progetto. **Verifica**: conteggio dei punti indicizzati in
`kb_docs`.

**RF-10** — Ogni chunk di policy DEVE conservare il titolo del documento di
provenienza, così da risultare comprensibile anche isolato dal resto del testo.
**Fonte**: progetto. **Verifica**: ispezione del contenuto indicizzato.

**RF-11** — Il sistema DEVE indicizzare lo storico dei ticket in una
collection distinta da quella delle policy, con un punto per ticket.
**Fonte**: progetto. **Verifica**: conteggio dei punti in `kb_tickets`.

**RF-12** — Per ogni ticket storico, il sistema DEVE calcolare l'embedding sul
solo enunciato del problema (oggetto e descrizione), mantenendo la risoluzione
nel payload associato.
**Fonte**: progetto. **Verifica**: ispezione del codice di indicizzazione e
del payload restituito.

**RF-13** — Il sistema DEVE interrogare entrambe le collection per ogni
richiesta ricevuta, prima di generare la risposta.
**Fonte**: POL-006 §4, che presuppone il recupero da entrambe le basi.
**Verifica**: tracce di esecuzione; presenza di due span di recupero.

**RF-14** — Le due interrogazioni DOVREBBERO essere eseguite in parallelo.
**Fonte**: progetto. **Verifica**: ispezione della struttura del workflow.

**RF-15** — Il sistema DEVE utilizzare rappresentazioni vettoriali distinte
per i documenti indicizzati e per le interrogazioni.
**Fonte**: progetto. **Verifica**: ispezione dei parametri di embedding.

**RF-16** — Il numero di risultati recuperati da ciascuna collection DEVE
essere configurabile senza modifiche al codice.
**Fonte**: progetto. **Verifica**: variazione del parametro e conteggio dei
risultati.

**RF-17** — In caso di errore nel recupero, il sistema DEVE proseguire con un
contesto vuoto anziché interrompere l'elaborazione della richiesta.
**Fonte**: progetto. **Verifica**: simulazione di indisponibilità del database
vettoriale; il turno si conclude comunque.

**RF-18** — Il sistema DEVE consentire di escludere specifici documenti dai
risultati di una interrogazione.
**Fonte**: progetto (necessario per la valutazione in leave-one-out).
**Verifica**: interrogazione con esclusione; il documento escluso non compare.

## Generazione della risposta

**RF-19** — Il sistema DEVE generare una bozza di risposta fondata sul
contesto recuperato dalle due basi di conoscenza.
**Fonte**: progetto. **Verifica**: valutazione di *groundedness* sulla suite
delle risposte.

**RF-20** — Quando il contesto recuperato non copre la situazione descritta,
il sistema DEVE dichiararlo esplicitamente anziché formulare procedure
aziendali non documentate.
**Fonte**: POL-006 §4. **Verifica**: casi di valutazione fuori dominio;
giudizio di *groundedness*.

**RF-21** — Il sistema DEVE produrre, insieme alla risposta, un punteggio di
confidenza nell'intervallo [0, 1].
**Fonte**: POL-006 §4, che vi fonda un criterio di escalation.
**Verifica**: ispezione della risposta dell'API.

**RF-22** — Il sistema DEVE estrarre dalla richiesta un insieme di
osservazioni strutturate: categoria, sottocategoria, priorità, e gli indicatori
richiamati dai criteri di escalation obbligatoria.
**Fonte**: POL-006 §3. **Verifica**: ispezione dello schema di risposta;
accuratezza misurata dalla suite di escalation.

**RF-23** — Le osservazioni strutturate DEVONO essere prodotte secondo uno
schema imposto al modello, tale da garantirne la conformità sintattica.
**Fonte**: progetto. **Verifica**: assenza di errori di interpretazione della
risposta su un campione di esecuzioni.

**RF-24** — Il modello NON DEVE disporre di alcun mezzo per dichiarare
direttamente la necessità di escalation: deve limitarsi a riportare quanto la
richiesta afferma.
**Fonte**: POL-006 §3, che impone alcune escalation indipendentemente dal
giudizio dell'agente. **Verifica**: ispezione dello schema; assenza di campi
decisionali.

**RF-25** — La risposta generata DOVREBBE essere autosufficiente, ossia
contenere quanto necessario a risolvere il problema senza ulteriori scambi:
passi ordinati e azionabili, assunzioni dichiarate quando manca un dettaglio,
criterio per riconoscere l'avvenuta risoluzione.
**Fonte**: progetto. **Verifica**: giudizio di pertinenza e completezza sulla
suite delle risposte.

**RF-26** — Il perseguimento dell'autosufficienza NON DEVE alterare il
punteggio di confidenza, che continua a misurare l'affidabilità della risposta.
**Fonte**: progetto. **Verifica**: confronto delle distribuzioni di confidenza
prima e dopo la modifica del prompt.

**RF-27** — In caso di errore nella generazione, il sistema DEVE restituire
una risposta di ripiego con confidenza nulla, tale da determinare
necessariamente l'escalation.
**Fonte**: progetto (principio di fallimento sicuro). **Verifica**:
simulazione di errore del provider; il ticket risulta escalato.

## Decisione di escalation

### Trigger obbligatori

I requisiti seguenti descrivono condizioni che impongono l'escalation
**indipendentemente dal punteggio di confidenza**, come prescritto da POL-006
§3. Nessuno di essi può essere annullato da una valutazione del modello.

**RF-28** — Il sistema DEVE escalare ogni ticket classificato nella categoria
*Security*.
**Fonte**: POL-005 §8, POL-006 §3.1. **Verifica**: caso ESC-015 della suite
di escalation, e ticket storici di categoria Security.

**RF-29** — Il sistema DEVE escalare ogni richiesta di cessazione involontaria
del rapporto di lavoro.
**Fonte**: POL-002 §6, POL-006 §3.2. **Verifica**: caso ESC-010.

**RF-30** — Il sistema DEVE escalare le richieste di accesso a sistemi
classificati come sensibili quando la documentazione delle approvazioni
richieste risulti incompleta.
**Fonte**: POL-002 §4, POL-006 §3.3. **Verifica**: caso ESC-009.

**RF-31** — Il sistema DEVE escalare le richieste che superano le soglie di
spesa definite, in assenza di approvazione documentata.
**Fonte**: POL-003 §5, POL-004 §4, POL-006 §3.4. **Verifica**: suite di
escalation sui ticket storici corrispondenti.

**RF-32** — Il sistema DEVE escalare le richieste di installazione di software
non presente nel catalogo approvato, senza prometterne l'esito.
**Fonte**: POL-004 §3. **Verifica**: suite di escalation sui ticket storici
corrispondenti.

**RF-33** — Il sistema DEVE escalare quando il richiedente chieda
esplicitamente di parlare con una persona, dichiari che l'assistenza
automatica non è efficace, o rifiuti la risoluzione proposta.
**Fonte**: POL-006 §3.5. **Verifica**: casi ESC-006, ESC-007, ESC-008.

**RF-34** — Il sistema DEVE escalare quando la richiesta presenti indizi di
impatto su più utenti o sull'infrastruttura.
**Fonte**: POL-006 §3.7, POL-007 §3. **Verifica**: suite di escalation sui
ticket storici corrispondenti.

**RF-35** — Il sistema DEVE escalare, indirizzandole alla funzione competente,
le richieste che esulano dall'ambito del supporto informatico: segnalazioni
riguardanti la condotta di colleghi, questioni retributive, contenziosi legali,
richieste di dati personali altrui.
**Fonte**: POL-008 §3, POL-008 §5. **Verifica**: casi da ESC-001 a ESC-004.

**RF-36** — Il sistema DEVE escalare ogni richiesta di aggirare un requisito
di approvazione previsto da una policy.
**Fonte**: POL-008 §5. **Verifica**: caso ESC-005.

### Trigger basati su confidenza e recupero

**RF-37** — Il sistema DEVE escalare quando il punteggio di confidenza risulti
inferiore alla soglia configurata.
**Fonte**: POL-006 §4, che indica il valore 0,65. **Verifica**: casi ESC-011 e
ESC-012.

**RF-38** — Il sistema DEVE escalare quando nessun risultato, **né** tra le
policy **né** tra i ticket storici, superi la soglia minima di similarità: in
tal caso non esiste base documentale per una risposta automatica.
**Fonte**: POL-006 §4. **Verifica**: casi ESC-013 e ESC-014.

**RF-39** — Il sistema DOVREBBE escalare quando, tra i ticket storici
sufficientemente simili alla richiesta, la maggioranza risulti a suo tempo
escalata.
**Fonte**: POL-006 §6, che riconosce ai ticket risolti valore di precedente.
**Verifica**: suite di escalation; contributo del trigger corrispondente.

### Combinazione e tracciabilità della decisione

**RF-40** — Il sistema DEVE escalare il ticket se **almeno una** delle
condizioni previste dai requisiti RF-28..RF-39 risulta soddisfatta.
**Fonte**: progetto (scelta conservativa motivata dall'asimmetria dei costi
d'errore). **Verifica**: ispezione della logica di combinazione.

**RF-41** — Il sistema DEVE registrare **tutte** le condizioni soddisfatte,
non soltanto la prima, e per ciascuna DEVE indicare la clausola di policy che
la giustifica.
**Fonte**: POL-006 §5.2. **Verifica**: ispezione dell'elenco restituito;
misura dell'accuratezza per singolo trigger.

**RF-42** — Le soglie numeriche che governano la decisione DEVONO essere
configurabili senza modifiche al codice.
**Fonte**: progetto. **Verifica**: variazione dei parametri di configurazione e
osservazione del comportamento.

**RF-43** — La decisione di escalation DEVE essere presa da logica
deterministica e ispezionabile, distinta dal componente che genera la risposta.
**Fonte**: POL-006 §3. **Verifica**: ispezione dell'architettura; il modulo
decisionale non invoca il modello linguistico.

## Flusso di revisione umana

**RF-44** — Quando la decisione impone l'escalation, il sistema DEVE
sospendere l'elaborazione prima di consegnare la risposta al richiedente.
**Fonte**: POL-005 §2, POL-006 §5.1. **Verifica**: prova end-to-end; la
risposta non raggiunge il richiedente prima della revisione.

**RF-45** — Lo stato dell'elaborazione sospesa DEVE essere conservato in modo
da poter essere ripreso da una richiesta successiva e indipendente, senza che
alcun processo resti in attesa.
**Fonte**: progetto. **Verifica**: revisione effettuata a distanza di tempo
dall'apertura del ticket.

**RF-46** — Il ticket sospeso DEVE essere inserito in una coda consultabile
dagli operatori.
**Fonte**: POL-006 §5.1. **Verifica**: `GET /api/tickets`.

**RF-47** — Il richiedente DEVE essere informato che la richiesta è stata
inoltrata a un operatore.
**Fonte**: POL-006 §5.3. **Verifica**: prova funzionale sull'interfaccia
utente.

**RF-48** — Il richiedente NON DEVE poter modificare la bozza prodotta
dall'agente.
**Fonte**: progetto. **Verifica**: assenza di comandi di modifica
nell'interfaccia utente e di endpoint corrispondenti.

**RF-49** — All'operatore DEVE essere presentato il contesto completo del
ticket: richiesta originale, cronologia, bozza generata, confidenza dichiarata
ed elenco dei trigger scattati.
**Fonte**: POL-006 §5.2. **Verifica**: ispezione della console operatore.

**RF-50** — L'operatore DEVE poter modificare liberamente la bozza oppure
approvarla senza modifiche.
**Fonte**: POL-006 §5.2. **Verifica**: prova funzionale.

**RF-51** — All'invio da parte dell'operatore, il sistema DEVE riprendere
l'elaborazione sospesa e consolidare il testo fornito come risposta definitiva.
**Fonte**: progetto. **Verifica**: prova end-to-end.

**RF-52** — Il sistema DEVE rifiutare un tentativo di revisione su una
conversazione priva di elaborazione sospesa.
**Fonte**: progetto. **Verifica**: `POST /api/review` su conversazione già
conclusa restituisce errore 400.

**RF-53** — Alla risoluzione, il ticket DEVE essere rimosso dalla coda degli
operatori.
**Fonte**: progetto. **Verifica**: `GET /api/tickets` successivo alla
risoluzione.

**RF-54** — La risposta consegnata al richiedente DEVE indicare se è stata
verificata da un operatore.
**Fonte**: progetto. **Verifica**: ispezione dell'interfaccia utente.

**RF-55** — Il richiedente DEVE ricevere la risposta senza dover ricaricare la
pagina né intraprendere alcuna azione.
**Fonte**: progetto. **Verifica**: prova end-to-end su due schede del browser.

## Popolamento della base di conoscenza

**RF-56** — Il sistema DEVE creare le collection e popolarle quando queste non
esistano.
**Fonte**: progetto. **Verifica**: primo avvio su volume vuoto.

**RF-57** — Quando le collection esistono già, il sistema DEVE aggiornarle
anziché ricostruirle.
**Fonte**: progetto. **Verifica**: secondo avvio; conteggio delle operazioni di
scrittura.

**RF-58** — L'aggiornamento DEVE essere incrementale: devono essere ricalcolati
i soli elementi nuovi o modificati.
**Fonte**: progetto. **Verifica**: riavvio senza modifiche; nessuna chiamata al
servizio di embedding.

**RF-59** — Gli elementi rimossi dalla sorgente DEVONO essere eliminati
dall'indice.
**Fonte**: progetto. **Verifica**: rimozione di un ticket dal file sorgente e
successiva sincronizzazione.

**RF-60** — Il sistema DEVE rilevare l'incompatibilità tra un indice esistente
e la configurazione di embedding attiva, e in tal caso ricostruire l'indice.
**Fonte**: progetto. **Verifica**: variazione della dimensione dei vettori e
osservazione della ricostruzione.

**RF-61** — Il popolamento DEVE completarsi con successo prima che il sistema
accetti richieste dagli utenti.
**Fonte**: progetto. **Verifica**: ispezione delle dipendenze di avvio dei
servizi.

## Osservabilità e valutazione

**RF-62** — Il sistema DEVE registrare una traccia per ogni elaborazione,
articolata negli stadi che la compongono.
**Fonte**: progetto. **Verifica**: consultazione del servizio di tracciamento.

**RF-63** — Le tracce DEVONO includere i risultati del recupero con i relativi
punteggi e l'esito della decisione di escalation.
**Fonte**: progetto. **Verifica**: ispezione del contenuto di una traccia.

**RF-64** — Il sistema DEVE registrare la configurazione attiva insieme alle
tracce, così da rendere confrontabili esecuzioni ottenute con parametri
diversi.
**Fonte**: progetto. **Verifica**: ispezione dei parametri registrati.

**RF-65** — Il tracciamento NON DEVE compromettere il servizio: se il
sottosistema di osservabilità non è disponibile, l'elaborazione deve
proseguire.
**Fonte**: progetto. **Verifica**: esecuzione con servizio di tracciamento
spento.

**RF-66** — Le credenziali NON DEVONO comparire in alcun registro o
parametro tracciato.
**Fonte**: POL-008 §2. **Verifica**: ispezione dei registri e dei parametri
registrati.

**RF-67** — Il sistema DEVE fornire una modalità di valutazione della qualità
del recupero, in leave-one-out sui ticket storici.
**Fonte**: progetto. **Verifica**: esecuzione della suite corrispondente.

**RF-68** — Il sistema DEVE fornire una modalità di valutazione
dell'accuratezza della decisione di escalation, distinguendo i casi tratti
dallo storico da quelli costruiti appositamente.
**Fonte**: progetto. **Verifica**: esecuzione delle due suite.

**RF-69** — Il sistema DEVE fornire una modalità di valutazione della qualità
delle risposte generate, con criteri di fondatezza, pertinenza e conformità
alle policy.
**Fonte**: progetto. **Verifica**: esecuzione della suite corrispondente.

**RF-70** — Le valutazioni DEVONO produrre, oltre alle metriche aggregate, il
dettaglio per singolo caso.
**Fonte**: progetto. **Verifica**: presenza della tabella dei risultati tra gli
artefatti del run.

**RF-71** — Il sistema DEVE fornire uno strumento di calibrazione delle soglie
che non impieghi i casi di valutazione etichettati.
**Fonte**: progetto (prevenzione dell'adattamento ai dati di test).
**Verifica**: ispezione dei dati impiegati dallo strumento.

\newpage

# Requisiti non funzionali

Trattati in forma sintetica, come previsto dallo scopo del documento.

**RNF-01 (Affidabilità)** — Il sistema DEVE degradare in modo controllato: il
guasto di un componente accessorio non deve impedire il completamento di una
richiesta. In particolare, l'indisponibilità del recupero comporta un contesto
vuoto, quella del modello una risposta di ripiego a confidenza nulla, quella
del tracciamento nessun effetto sul servizio.

**RNF-02 (Sicurezza)** — Le credenziali DEVONO essere fornite tramite
configurazione d'ambiente, non devono essere versionate né comparire nei
registri. Il sistema DEVE segnalare all'avvio la presenza e la validità
formale della credenziale attiva.

**RNF-03 (Prestazioni)** — Il tempo di attesa percepito dal richiedente per la
consegna di una risposta revisionata NON DEVE superare i 5 secondi dal momento
dell'invio da parte dell'operatore.

**RNF-04 (Manutenibilità)** — I parametri che governano il comportamento del
sistema DEVONO risiedere in configurazione centralizzata, non nel codice.

**RNF-05 (Portabilità)** — Il sistema DEVE essere avviabile su una singola
macchina con un unico comando, senza installazione manuale di dipendenze.

**RNF-06 (Verificabilità)** — Il richiamo alla classe *escalation* DEVE essere
almeno pari a 0,90. La *precisione* sulla stessa classe DOVREBBE essere almeno
pari a 0,60. La scelta di privilegiare il richiamo discende dall'asimmetria dei
costi: la mancata escalation di un ticket che la richiedeva ha conseguenze
sostanzialmente più gravi dell'escalation di un ticket risolvibile.

**RNF-07 (Usabilità)** — L'interfaccia dell'operatore DEVE presentare i motivi
dell'escalation in forma leggibile, con esplicito riferimento alla clausola di
policy applicata.

\newpage

# Matrice di tracciabilità

Riepilogo della corrispondenza tra requisiti, origine e modalità di verifica.

| Requisiti | Origine | Componente | Verifica |
|---|---|---|---|
| RF-01..RF-08 | progetto | interfaccia HTTP, registro conversazioni | prove funzionali |
| RF-09..RF-18 | progetto, POL-006 §4 | sottosistema di recupero | suite *retrieval* |
| RF-19..RF-27 | progetto, POL-006 §4 | sottosistema di generazione | suite *answers* |
| RF-28..RF-36 | POL-002..POL-008 | motore di regole | suite *escalation* |
| RF-37..RF-39 | POL-006 §4, §6 | motore di regole | suite *escalation* |
| RF-40..RF-43 | POL-006 §3, §5.2 | motore di regole | ispezione e suite |
| RF-44..RF-55 | POL-005 §2, POL-006 §5 | workflow, console operatore | prove end-to-end |
| RF-56..RF-61 | progetto | procedura di popolamento | prove di avvio |
| RF-62..RF-71 | progetto, POL-008 §2 | osservabilità e valutazione | esecuzione delle suite |

\newpage

# Appendice A — Struttura di un ticket storico

| Campo | Tipo | Descrizione |
|---|---|---|
| `ticket_id` | stringa | Identificativo univoco |
| `created_at` | data | Data di apertura |
| `source_channel` | stringa | Canale di provenienza |
| `department` | stringa | Reparto del richiedente |
| `requester_role` | stringa | Ruolo del richiedente |
| `category` | enumerazione | Categoria (7 valori) |
| `subcategory` | stringa | Sottocategoria (20 valori) |
| `priority` | enumerazione | Da P1 a P4 |
| `subject` | stringa | Oggetto della richiesta |
| `description` | testo | Descrizione del problema |
| `resolution_steps` | elenco | Passi seguiti per la risoluzione |
| `resolution_summary` | testo | Sintesi della risoluzione |
| `resolution_time_minutes` | intero | Tempo impiegato |
| `status` | enumerazione | Stato finale |
| `was_escalated_to_human` | booleano | **Etichetta di riferimento per la valutazione** |
| `escalation_reason` | testo | Motivazione dell'escalation |
| `csat_score` | intero | Gradimento espresso, ove disponibile |
| `tags` | elenco | Parole chiave |

# Appendice B — Riepilogo dei trigger di escalation

| Trigger | Requisito | Clausola |
|---|---|---|
| Categoria *Security* | RF-28 | POL-005 §8, POL-006 §3.1 |
| Cessazione involontaria | RF-29 | POL-006 §3.2 |
| Accesso sensibile senza approvazioni | RF-30 | POL-006 §3.3 |
| Superamento soglia di spesa | RF-31 | POL-006 §3.4 |
| Software fuori catalogo | RF-32 | POL-004 §3 |
| Richiesta esplicita di un operatore | RF-33 | POL-006 §3.5 |
| Impatto multi-utente | RF-34 | POL-006 §3.7 |
| Richiesta fuori ambito | RF-35 | POL-008 §5 |
| Elusione di un'approvazione | RF-36 | POL-008 §5 |
| Confidenza sotto soglia | RF-37 | POL-006 §4 |
| Assenza di base documentale | RF-38 | POL-006 §4 |
| Precedenti storici escalati | RF-39 | POL-006 §6 |

# Appendice C — Riferimento ai diagrammi

| Diagramma | Requisiti illustrati |
|---|---|
| Casi d'uso operativi | RF-01..RF-08, RF-44..RF-55 |
| Casi d'uso di amministrazione | RF-56..RF-71 |
| Componenti | RNF-05, RF-61 |
| Classi (vista astratta) | RF-43 |
| Classi (vista dettagliata) | RF-22, RF-24, RF-41 |
| Sequenza del flusso HITL | RF-44..RF-55 |
| Attività della decisione | RF-28..RF-43 |
