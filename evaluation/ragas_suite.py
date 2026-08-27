"""
evaluation/ragas_suite.py
=========================
Valutazione della pipeline RAG con **RAGAS**, con risultati registrati su
MLflow.

    python -m evaluation.ragas_suite --sample 20

Perché affiancare RAGAS al giudice già presente
-----------------------------------------------
`evaluation/judge.py` misura già fondatezza, pertinenza e conformità alle
policy con un prompt scritto su misura. RAGAS non lo sostituisce: aggiunge
metriche **standardizzate e citabili**, calcolate con procedure pubblicate e
identiche a quelle usate in letteratura. Le due cose rispondono a domande
diverse:

- il giudice risponde a *"questa risposta rispetta le nostre policy?"*, che
  nessuna libreria generica può sapere;
- RAGAS risponde a *"questa pipeline RAG è buona secondo criteri riconosciuti,
  confrontabili con altri sistemi?"*.

Per una tesi la seconda ha un valore specifico: permette di dire "faithfulness
0,82" con un significato che un lettore esterno può interpretare senza leggere
il nostro prompt di giudizio.

Le quattro metriche adottate
----------------------------
La scelta separa deliberatamente ciò che misura il **recupero** da ciò che
misura la **generazione**, perché sono i due stadi che possono guastarsi
indipendentemente e richiedono rimedi diversi.

| Metrica | Stadio | Domanda a cui risponde |
|---|---|---|
| `ContextPrecisionWithoutReference` | recupero | i chunk recuperati sono pertinenti alla richiesta? |
| `ContextRecall` | recupero | il contesto recuperato copre quanto serve per rispondere? |
| `Faithfulness` | generazione | ogni affermazione della risposta è sostenuta dal contesto? |
| `ResponseRelevancy` | generazione | la risposta affronta davvero la domanda posta? |

`ContextRecall` è l'unica che richiede un riferimento. Si usa
`resolution_summary` del ticket storico, ed è un impiego legittimo: **non si
confronta la risposta con il riferimento**, si verifica se il contesto
recuperato contiene le informazioni presenti nel riferimento. La distinzione
conta, perché altrove in questo progetto si è scelto di *non* usare
`resolution_summary` come riferimento per la correttezza della risposta: è una
nota interna scritta a posteriori, con destinatario e registro diversi da una
risposta all'utente. Per la copertura del contesto, invece, quello stesso testo
è un buon indicatore di ciò che il recupero avrebbe dovuto trovare.

Costo
-----
Ogni metrica comporta una o più chiamate al modello per campione: valutare 20
casi significa nell'ordine di un centinaio di chiamate. Il campionamento è
quindi il comportamento predefinito, non un'opzione.

Dipendenze
----------
RAGAS non è fra le dipendenze principali del progetto ma in un extra
opzionale, perché richiede `langchain-community<0.4`: le versioni successive
hanno rimosso un modulo che RAGAS importa ancora, e allinearsi a quel vincolo
nell'ambiente principale significherebbe condizionare il servizio a una
libreria che serve solo in valutazione.

    uv sync --extra ragas
"""
import argparse
import logging
from typing import Any, Dict, List

from app import config
from evaluation.run_evaluation import run_system, stratified_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ragas-suite")


def costruisci_dataset(casi: List[Dict[str, Any]]) -> Any:
    """
    Esegue il sistema su ogni caso e ne raccoglie gli esiti nel formato RAGAS.

    Ogni campione porta quattro elementi: la richiesta, i contesti
    effettivamente recuperati, la risposta prodotta e il riferimento.

    I contesti sono passati come **elenco di testi separati**, non concatenati:
    `ContextPrecision` valuta la pertinenza di ciascun chunk singolarmente e
    ne misura anche la posizione nel ranking. Concatenarli in un unico blocco
    renderebbe la metrica cieca a quale chunk sia utile e quale no, che è
    proprio l'informazione che serve per capire se il ranking funziona.

    Il leave-one-out è applicato come nelle altre suite: un ticket usato come
    query non deve recuperare sé stesso, altrimenti tutte le metriche di
    recupero risulterebbero perfette per costruzione.
    """
    from ragas import EvaluationDataset, SingleTurnSample

    campioni = []
    for c in casi:
        esito = run_system(c["query"], exclude_sources=[c["case_id"]])
        contesti = [d.get("text", "") for d in esito["kb_docs"]] + \
                   [t.get("text", "") for t in esito["kb_tickets"]]

        campioni.append(SingleTurnSample(
            user_input=c["query"],
            retrieved_contexts=contesti,
            response=esito["answer"],
            reference=c["reference"],
        ))
        logger.info("Raccolto %s (%d contesti, escalato=%s)",
                    c["case_id"], len(contesti), esito["escalated"])

    return EvaluationDataset(samples=campioni)


def costruisci_modelli():
    """
    Fornisce a RAGAS il modello e gli embedding da usare come giudice.

    RAGAS non ha un modello proprio: ogni metrica è una procedura che
    interroga un LLM. Si passa quindi lo stesso provider usato dal sistema,
    tramite gli adattatori LangChain che RAGAS si aspetta.

    Nota metodologica: usare per la valutazione lo stesso modello che genera
    le risposte riduce l'indipendenza del giudizio, perché valutato e
    valutatore condividono gli stessi punti ciechi. `JUDGE_MODEL` è
    configurabile proprio per poterli separare, ed è la prima cosa da variare
    se i punteggi appaiono sistematicamente generosi.
    """
    from google import genai
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory

    # Si riusa lo stesso client dell'applicazione: RAGAS lo accetta
    # direttamente, senza passare da adattatori LangChain.
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    llm = llm_factory(config.JUDGE_MODEL, provider="google", client=client)
    emb = GoogleEmbeddings(client=client, model=config.GEMINI_EMBEDDING_MODEL)
    return llm, emb


def esegui(sample: int = 20) -> Dict[str, float]:
    """
    Esegue la valutazione RAGAS e restituisce le metriche aggregate.

    Il campione è stratificato per sottocategoria con seme fisso, come nelle
    altre suite: due esecuzioni devono valutare gli stessi casi, altrimenti le
    differenze fra un run e l'altro sarebbero rumore indistinguibile da un
    effetto reale.
    """
    from ragas import evaluate
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithoutReference,
        ContextRecall,
        Faithfulness,
    )

    if not config.GEMINI_API_KEY:
        logger.error("Serve una chiave API: le metriche RAGAS sono calcolate da un LLM.")
        return {}

    casi = stratified_sample(load_ticket_cases_con_riferimento(), sample)
    dataset = costruisci_dataset(casi)
    llm, emb = costruisci_modelli()

    # Nelle metriche "collections" il giudice si passa al COSTRUTTORE, non a
    # evaluate(): ogni metrica incapsula il proprio modello, il che permette
    # in linea di principio di usarne di diversi per criteri diversi.
    risultato = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecisionWithoutReference(llm=llm),
            ContextRecall(llm=llm),
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm, embeddings=emb),
        ],
        # Un fallimento su un singolo campione non interrompe la valutazione:
        # il caso riceve NaN ed è escluso dalla media.
        raise_exceptions=False,
    )

    # Le medie si calcolano dal dataframe dei risultati per caso invece di
    # leggere attributi interni dell'oggetto risultato, che cambiano fra
    # versioni di RAGAS. `mean()` ignora i NaN, quindi i campioni falliti non
    # falsano la media: `n_valid` dice su quanti è stata calcolata davvero.
    df = risultato.to_pandas()
    metriche: Dict[str, float] = {}
    for colonna in df.select_dtypes(include="number").columns:
        metriche[f"ragas/{colonna}"] = float(df[colonna].mean())
        metriche[f"ragas/{colonna}_n_valid"] = float(df[colonna].notna().sum())
    metriche["ragas/n_samples"] = float(len(casi))
    return metriche, risultato


def load_ticket_cases_con_riferimento() -> List[Dict[str, Any]]:
    """
    Come `load_ticket_cases`, ma aggiunge il riferimento richiesto da
    `ContextRecall`.

    Il riferimento è `resolution_summary`: la sintesi di come il ticket fu
    effettivamente risolto. Serve a stabilire se il contesto recuperato
    contiene le informazioni necessarie, non a giudicare la formulazione della
    risposta.
    """
    import json
    from pathlib import Path

    kb = Path(__file__).parent.parent / "app" / "knowledge_base" / "past_tickets.json"
    tickets = json.loads(kb.read_text(encoding="utf-8"))
    return [
        {
            "case_id": t["ticket_id"],
            "query": f"{t['subject']}\n\n{t['description']}",
            "subcategory": t["subcategory"],
            "reference": t["resolution_summary"],
        }
        for t in tickets
    ]


def registra_su_mlflow(metriche: Dict[str, float], risultato: Any) -> None:
    """
    Registra metriche, parametri e dettaglio per caso in un unico run MLflow.

    Il dettaglio per caso è salvato come tabella: una media di faithfulness
    dice che qualcosa non va, la tabella mostra su quali richieste. Con
    metriche prodotte da un LLM è ancora più importante, perché un punteggio
    anomalo può dipendere tanto dal sistema quanto da un giudizio sbagliato, e
    solo leggendo il caso si distingue.
    """
    try:
        import mlflow

        from app import prompts

        prompts.register_agent_prompt()
        mlflow.set_experiment(f"{config.MLFLOW_EXPERIMENT}-eval")
        with mlflow.start_run(run_name="eval-ragas"):
            mlflow.log_params(config.as_params())
            mlflow.log_params(prompts.as_params())
            mlflow.log_param("evaluation_framework", "ragas")
            mlflow.log_metrics(metriche)

            df = risultato.to_pandas()
            mlflow.log_table(data=df, artifact_file="ragas_per_case.json")
            logger.info("Run RAGAS registrato su MLflow.")
    except Exception:
        logger.exception("Registrazione su MLflow fallita: le metriche restano a video.")


def main() -> int:
    """Entrypoint da riga di comando."""
    parser = argparse.ArgumentParser(description="Valutazione RAG con RAGAS")
    parser.add_argument("--sample", type=int, default=20,
                        help="numero di ticket da valutare (ogni caso costa più chiamate LLM)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    esito = esegui(args.sample)
    if not esito:
        return 1
    metriche, risultato = esito

    print("\n=== Metriche RAGAS ===")
    for nome, valore in sorted(metriche.items()):
        print(f"  {nome:40s} {valore:.4f}")

    if not args.no_mlflow:
        registra_su_mlflow(metriche, risultato)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
