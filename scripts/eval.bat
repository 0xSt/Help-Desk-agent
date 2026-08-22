@echo off
REM ===================================================================
REM eval.bat - lancia l'evaluation su Windows (Prompt dei comandi)
REM
REM Uso, dalla root del progetto:
REM     scripts\eval.bat calibrate
REM     scripts\eval.bat retrieval
REM     scripts\eval.bat escalation
REM     scripts\eval.bat answers
REM     scripts\eval.bat all
REM
REM Carica le variabili dal file .env, che su Windows Python NON legge da
REM solo, e poi imposta i puntamenti ai servizi Docker esposti sull'host.
REM
REM ATTENZIONE: nel file .env i valori NON devono essere tra virgolette.
REM Docker Compose le toglie da solo, il ciclo qui sotto no: la chiave
REM verrebbe passata con le virgolette incluse e l'API la rifiuterebbe.
REM Scrivi  GEMINI_API_KEY=AIza...   non  GEMINI_API_KEY="AIza..."
REM ===================================================================

setlocal enabledelayedexpansion

if not exist ".env" (
    echo [ERRORE] File .env non trovato nella cartella corrente.
    echo Lancia lo script dalla root del progetto, dove sta docker-compose.yml.
    exit /b 1
)

REM eol=# salta le righe di commento; delims== separa nome e valore
for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
    if not "%%a"=="" set "%%a=%%b"
)

REM Puntamenti ai servizi containerizzati, esposti sull'host dal compose
set "QDRANT_URL=http://localhost:6333"
set "MLFLOW_TRACKING_URI=http://localhost:5000"
REM L'indice lo popola il job di ingestion, non l'evaluation
set "AUTO_INDEX=false"

if "%GEMINI_API_KEY%"=="" (
    echo [ATTENZIONE] GEMINI_API_KEY risulta vuota: l'evaluation girerebbe
    echo in modalita' mock e i numeri non direbbero nulla sul sistema reale.
    exit /b 1
)

set "COMANDO=%~1"
if "%COMANDO%"=="" set "COMANDO=all"

if /i "%COMANDO%"=="calibrate" (
    python -m evaluation.calibrate_thresholds
) else (
    python -m evaluation.run_evaluation --suite %COMANDO% --sample 30
)

endlocal
