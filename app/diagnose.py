"""
diagnose.py
===========
Diagnosi della connessione alle API Gemini.

    python -m app.diagnose

Dentro Docker Compose (importante: eseguirlo **nel container**, perché è lì
che il problema si manifesta e l'ambiente è diverso da quello dell'host):

    docker compose run --rm ingestion python -m app.diagnose

Esegue sei controlli in sequenza, dal più basso al più alto livello, e si
ferma al primo che fallisce. Il punto è isolare *quale* strato è rotto: una
chiave assente, un DNS che non risolve, un firewall aziendale, un nome di
modello inesistente e una quota esaurita producono tutti lo stesso sintomo
apparente ("non funziona con Gemini") ma richiedono rimedi completamente
diversi.
"""
import logging
import os
import socket
import sys

from app import config

logging.basicConfig(level=logging.WARNING)

API_HOST = "generativelanguage.googleapis.com"

OK = "  [OK]  "
KO = "  [KO]  "
INFO = "        "


def _hint(text: str) -> None:
    for line in text.strip().splitlines():
        print(f"{INFO}-> {line.strip()}")


def check_1_credenziali() -> bool:
    print("\n1. CREDENZIALI")
    print(f"{INFO}{config.describe_credentials()}")
    if not config.GEMINI_API_KEY:
        print(KO + "Nessuna chiave attiva.")
        _hint("""
            In Docker: verifica che .env sia nella stessa cartella del
            docker-compose.yml e che la riga sia GEMINI_API_KEY=AIza...
            senza virgolette e senza spazi attorno all'uguale.
            In locale: il .env NON viene letto da solo, esporta le variabili.
        """)
        return False
    print(OK + "Chiave presente.")

    # Errore frequente: copiare la chiave con spazi o a capo invisibili.
    key = config.GEMINI_API_KEY
    if key != key.strip():
        print(KO + "La chiave contiene spazi o newline iniziali/finali.")
        _hint("Rimuovi spazi e virgolette dal valore nel file .env.")
        return False
    if key.startswith(('"', "'")) or key.endswith(('"', "'")):
        print(KO + "La chiave è racchiusa tra virgolette.")
        _hint("Nel file .env il valore va scritto senza virgolette.")
        return False
    return True


def check_2_sdk() -> bool:
    print("\n2. SDK")
    try:
        from google import genai
        print(OK + f"google-genai importato (versione {getattr(genai, '__version__', 'n/d')}).")
        return True
    except ImportError as e:
        print(KO + f"Import fallito: {e}")
        _hint("Ricostruisci l'immagine: docker compose build --no-cache")
        return False


def check_3_dns() -> bool:
    print("\n3. DNS")
    try:
        ip = socket.gethostbyname(API_HOST)
        print(OK + f"{API_HOST} risolve a {ip}.")
        return True
    except socket.gaierror as e:
        print(KO + f"Risoluzione DNS fallita: {e}")
        _hint("""
            Il container non riesce a risolvere il nome. Cause tipiche:
            DNS aziendale che non risponde dentro Docker, oppure assenza di
            connettività di rete nel container.
            Prova: docker compose run --rm ingestion getent hosts google.com
        """)
        return False


def check_4_tcp() -> bool:
    print("\n4. CONNETTIVITÀ TCP (porta 443)")
    try:
        with socket.create_connection((API_HOST, 443), timeout=10):
            print(OK + "Connessione TCP stabilita.")
    except Exception as e:
        print(KO + f"Connessione fallita: {type(e).__name__}: {e}")
        _hint("""
            DNS funziona ma la porta 443 non è raggiungibile: quasi sempre un
            firewall o un proxy aziendale.
            Se sei dietro un proxy, passa HTTPS_PROXY e HTTP_PROXY al
            container aggiungendoli al file .env.
        """)
        return False

    proxy = {k: v for k, v in os.environ.items() if k.lower() in
             ("http_proxy", "https_proxy", "no_proxy")}
    if proxy:
        print(f"{INFO}Proxy configurato: {proxy}")
    return True


def check_5_modelli() -> bool:
    print("\n5. MODELLI DISPONIBILI")
    from google import genai

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        modelli = list(client.models.list())
    except Exception as e:
        print(KO + f"Chiamata fallita: {type(e).__name__}: {e}")
        _diagnostica_errore_api(e)
        return False

    gen, emb = [], []
    for m in modelli:
        azioni = m.supported_actions or []
        if "generateContent" in azioni:
            gen.append(m.name)
        if "embedContent" in azioni:
            emb.append(m.name)

    print(OK + f"{len(modelli)} modelli visibili con questa chiave.")

    def _verifica(configurato: str, disponibili: list, etichetta: str) -> bool:
        # L'API restituisce nomi nella forma "models/xxx"; la configurazione
        # usa il nome nudo. Confrontiamo entrambe le forme.
        nomi = {n.split("/")[-1] for n in disponibili}
        if configurato in nomi:
            print(OK + f"{etichetta} configurato '{configurato}': disponibile.")
            return True
        print(KO + f"{etichetta} configurato '{configurato}': NON disponibile.")
        simili = sorted(n for n in nomi if configurato.split("-")[0] in n)[:8]
        _hint("Valori validi per questa chiave, tra cui scegliere:\n"
              + "\n".join(f"  {n}" for n in (simili or sorted(nomi)[:8])))
        return False

    ok_gen = _verifica(config.GEMINI_MODEL, gen, "Modello di generazione")
    ok_emb = _verifica(config.GEMINI_EMBEDDING_MODEL, emb, "Modello di embedding")

    if not (ok_gen and ok_emb):
        _hint("Correggi GEMINI_MODEL / GEMINI_EMBEDDING_MODEL nel file .env.")
    return ok_gen and ok_emb


def check_6_chiamate() -> bool:
    print("\n6. CHIAMATE REALI")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    esito = True

    # --- embedding ---
    try:
        r = client.models.embed_content(
            model=config.GEMINI_EMBEDDING_MODEL,
            contents=["VPN connection test"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=config.EMBEDDING_DIM,
            ),
        )
        dim = len(r.embeddings[0].values)
        if dim == config.EMBEDDING_DIM:
            print(OK + f"Embedding riuscito: {dim} dimensioni, come configurato.")
        else:
            print(KO + f"Embedding riuscito ma di {dim} dimensioni invece di {config.EMBEDDING_DIM}.")
            _hint("Allinea EMBEDDING_DIM nel .env, poi ricostruisci l'indice.")
            esito = False
    except Exception as e:
        print(KO + f"Embedding fallito: {type(e).__name__}: {e}")
        _diagnostica_errore_api(e)
        esito = False

    # --- generazione con structured output ---
    try:
        from app.llm import DraftAnswer

        r = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents="A user cannot connect to the company VPN from a hotel. Answer briefly.",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DraftAnswer,
            ),
        )
        parsed = r.parsed
        print(OK + f"Generazione riuscita (confidenza dichiarata: {parsed.confidence:.2f}).")
    except Exception as e:
        print(KO + f"Generazione fallita: {type(e).__name__}: {e}")
        _diagnostica_errore_api(e)
        esito = False

    return esito


def _diagnostica_errore_api(e: Exception) -> None:
    """
    Traduce gli errori più comuni dell'API in un rimedio concreto.

    L'ordine dei controlli va dal più specifico al più generico: un blocco di
    rete in uscita restituisce anch'esso un 403, e se controllassimo prima
    "permission" verrebbe scambiato per un problema di permessi della chiave,
    mandando a cercare nel posto sbagliato.
    """
    testo = str(e).lower()
    if "not in allowlist" in testo or "egress" in testo:
        _hint("""
            La rete che ospita il container blocca il dominio in uscita: è una
            restrizione dell'ambiente (proxy, firewall, policy di rete), non
            un problema del codice né della chiave.
        """)
    elif "api_key_invalid" in testo or "api key not valid" in testo:
        _hint("""
            La chiave è formalmente arrivata ma il servizio la rifiuta.
            Rigenerala da Google AI Studio e verifica di non aver copiato
            caratteri di troppo.
        """)
    elif "permission" in testo or "403" in testo:
        _hint("""
            Chiave valida ma senza permessi: spesso l'API non è abilitata sul
            progetto Google Cloud associato, oppure la chiave ha restrizioni
            per IP o referrer che bloccano le chiamate dal container.
        """)
    elif "not found" in testo or "404" in testo:
        _hint("""
            Il modello indicato non esiste per questa chiave: vedi l'elenco
            del controllo 5 e correggi il .env.
        """)
    elif "quota" in testo or "429" in testo or "resource_exhausted" in testo:
        _hint("""
            Quota esaurita o troppe richieste. Sul piano gratuito il limite è
            per minuto: attendi e riprova. Se capita durante l'ingestion,
            riduci EMBEDDING_BATCH_SIZE nel .env.
        """)
    elif "deadline" in testo or "timeout" in testo:
        _hint("Timeout di rete: probabile proxy o connessione molto lenta.")


def main() -> int:
    print("=" * 66)
    print("DIAGNOSI CONNESSIONE API GEMINI")
    print("=" * 66)
    print(f"{INFO}modello generazione : {config.GEMINI_MODEL}")
    print(f"{INFO}modello embedding   : {config.GEMINI_EMBEDDING_MODEL}")
    print(f"{INFO}dimensione vettori  : {config.EMBEDDING_DIM}")

    for check in (check_1_credenziali, check_2_sdk, check_3_dns,
                  check_4_tcp, check_5_modelli, check_6_chiamate):
        if not check():
            print("\n" + "=" * 66)
            print("DIAGNOSI INTERROTTA: risolvi il punto qui sopra e rilancia.")
            print("=" * 66)
            return 1

    print("\n" + "=" * 66)
    print("TUTTI I CONTROLLI SUPERATI: l'integrazione con Gemini funziona.")
    print("Se l'indice era stato costruito col fallback, verrà ricostruito")
    print("automaticamente al prossimo avvio dell'ingestion.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
