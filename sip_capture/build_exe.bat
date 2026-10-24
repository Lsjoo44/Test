@echo off
rem Build SIPCapture.exe (GUI) and sipcap.exe (CLI) into dist\
cd /d "%~dp0"
python -m pip install -r requirements.txt pyinstaller || goto :err
python -m PyInstaller --noconfirm --onefile --windowed --name SIPCapture --collect-all scapy sipcap_gui.py || goto :err
python -m PyInstaller --noconfirm --onefile --console --name sipcap --collect-all scapy sipcap_cli.py || goto :err
if exist ffmpeg.exe copy /y ffmpeg.exe dist\ >nul
if exist bcg729.dll copy /y bcg729.dll dist\ >nul
echo.
echo Build OK: dist\SIPCapture.exe  dist\sipcap.exe
pause
exit /b 0
:err
echo Build FAILED
pause
exit /b 1
