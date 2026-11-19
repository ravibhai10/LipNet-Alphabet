@echo off
REM LipNet startup: keep the verified eight-word model when its runtime exists.
setlocal

cd /d "%~dp0"
if "%LIPNET_ACCESS_FILE%"=="" if exist ".access\access.json" set "LIPNET_ACCESS_FILE=%~dp0.access\access.json"
if "%LIPNET_CPU_THREADS%"=="" set "LIPNET_CPU_THREADS=1"
set "LIPNET_HOST=127.0.0.1"
set "LIPNET_PORT=8000"
if exist ".venv-avhubert-cpu\Scripts\python.exe" if exist "runs\word_accuracy\avhubert_miracl8_v1\head.pt" goto word_runtime
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

echo Starting the single-process LipNet backend on http://%LIPNET_HOST%:%LIPNET_PORT% ...
python scripts\run_backend.py --host %LIPNET_HOST% --port %LIPNET_PORT%
exit /b %errorlevel%

:word_runtime
set "LIPNET_WORD_BACKEND=avhubert"
set "LIPNET_AVHUBERT_WORD_HEAD=%~dp0runs\word_accuracy\avhubert_miracl8_v1\head.pt"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
".venv-avhubert-cpu\Scripts\python.exe" scripts\patch_fairseq_meta.py
if errorlevel 1 exit /b 1
echo Starting LipNet with AV-HuBERT (eight-word reference) on http://%LIPNET_HOST%:%LIPNET_PORT% ...
".venv-avhubert-cpu\Scripts\python.exe" scripts\run_backend.py --host %LIPNET_HOST% --port %LIPNET_PORT%
exit /b %errorlevel%
