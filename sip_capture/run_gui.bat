@echo off
cd /d "%~dp0"
where pythonw >nul 2>nul && (start "" pythonw sipcap_gui.py) || (python sipcap_gui.py)
