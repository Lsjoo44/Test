@echo off
cd /d "%~dp0"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. Next: install Npcap (https://npcap.com) and put ffmpeg.exe in this folder for G.729.
pause
