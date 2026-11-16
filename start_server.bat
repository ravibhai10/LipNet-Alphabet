@echo off
REM LipNet startup: keep the verified eight-word model when its runtime exists.
setlocal

cd /d "%~dp0"
if exist ".venv-avhubert-cpu\Scripts\python.exe" if exist "runs\word_accuracy\avhubert_miracl8_v1\head.pt" goto word_runtime
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat

echo Starting Lip Alphabet server on http://localhost:8000 ...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
exit /b %errorlevel%

:word_runtime
set "LIPNET_WORD_BACKEND=avhubert"
set "LIPNET_AVHUBERT_WORD_HEAD=%~dp0runs\word_accuracy\avhubert_miracl8_v1\head.pt"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
".venv-avhubert-cpu\Scripts\python.exe" scripts\patch_fairseq_meta.py
if errorlevel 1 exit /b 1
echo Starting LipNet with eight-word AV-HuBERT and separate 500-word research mode on http://localhost:8000 ...
".venv-avhubert-cpu\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
exit /b %errorlevel%
