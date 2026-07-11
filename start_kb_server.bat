@echo off
REM start_kb_server.bat — launch kb_server in a truly detached window
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com

cd /d E:\peS2o_kb_faiss

REM Find a free port
set PORT=8765
:checkport
netstat -ano | findstr ":%PORT% " >nul
if %errorlevel% equ 0 (
    set /a PORT+=1
    goto checkport
)

echo Using port %PORT%
echo Logs: E:\peS2o_kb_faiss\kb_server_%PORT%.log

REM Launch in a new console window that closes but keeps process alive
start "kb-server" /min cmd /c "py -3 E:\peS2o_kb_faiss\kb_server.py --port %PORT% ^> E:\peS2o_kb_faiss\kb_server_%PORT%.log 2^>^&1"

echo Started.
timeout /t 3 /nobreak >nul
netstat -ano | findstr ":%PORT% "
echo.
echo To stop:
echo   netstat -ano ^| findstr ":%PORT% "    (find PID)
echo   taskkill /F /PID ^<pid^>
pause