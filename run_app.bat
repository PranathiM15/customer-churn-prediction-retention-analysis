@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating project environment...
    py -m venv .venv
)
echo Installing or updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo Starting ChurnScope...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
