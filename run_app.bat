@echo off
echo Installing dependencies...
py -m pip install -r requirements.txt
echo.
echo Running Gradio Web Application...
py app.py
pause
