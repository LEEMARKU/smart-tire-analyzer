@echo off
REM Smart Tire Analyzer — Multi-Service Launcher (Windows)
REM This script starts the backend API server and frontend dev server in separate terminal windows

setlocal enabledelayedexpansion

color 0A
echo.
echo ========================================================
echo   SMART TIRE ANALYZER — Service Launcher
echo ========================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ and add it to your PATH
    pause
    exit /b 1
)

echo.
echo Choose an option:
echo.
echo [1] Start Backend Only (API server on port 8000)
echo [2] Start Frontend Only (Dev server on port 3000)
echo [3] Start Both Backend and Frontend
echo [4] Development Mode (Both with hot reload)
echo [5] Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto start_backend_only
if "%choice%"=="2" goto start_frontend_only
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto start_dev_mode
if "%choice%"=="5" goto exit_script
goto invalid_choice

:start_backend_only
echo.
echo [*] Starting Backend Server...
echo [*] Setting PYTHONPATH=backend
set PYTHONPATH=backend
cd /d "%~dp0"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
goto end

:start_frontend_only
echo.
echo [*] Starting Frontend Dev Server...
cd /d "%~dp0frontend"
npm install >nul 2>&1
npm run dev
pause
goto end

:start_both
echo.
echo [*] Starting Backend Server in new window...
start "Smart Tire Backend" cmd /k "cd /d %~dp0 & set PYTHONPATH=backend & python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 2 /nobreak

echo [*] Starting Frontend Dev Server in new window...
start "Smart Tire Frontend" cmd /k "cd /d %~dp0frontend & npm install >nul 2>&1 & npm run dev"
timeout /t 2 /nobreak

echo.
echo ========================================================
echo [✓] Both services started!
echo.
echo Backend:  http://localhost:8000 (API)
echo           http://localhost:8000/docs (Swagger UI)
echo           http://localhost:8000/redoc (ReDoc)
echo.
echo Frontend: http://localhost:3000 (Web App)
echo           http://localhost:3000/live-chat (Live Chat AI)
echo           http://localhost:3000/technical-support (Voice AI)
echo.
echo [*] Press Ctrl+C in each window to stop the service
echo ========================================================
echo.
pause
goto end

:start_dev_mode
echo.
echo [*] Starting in Development Mode (Full Debug)...
echo [*] Both services will run with hot-reload enabled
echo.
echo Starting Backend Server in new window...
start "Smart Tire Backend [DEBUG]" cmd /k "cd /d %~dp0 & set PYTHONPATH=backend & set DEBUG=1 & python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug"
timeout /t 2 /nobreak

echo Starting Frontend Dev Server in new window...
start "Smart Tire Frontend [DEBUG]" cmd /k "cd /d %~dp0frontend & npm install & npm run dev"
timeout /t 2 /nobreak

echo.
echo ========================================================
echo [✓] Development Mode Active!
echo.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:3000
echo.
echo Logs will display in each terminal window
echo ========================================================
echo.
pause
goto end

:invalid_choice
color 0C
echo.
echo ERROR: Invalid choice. Please enter 1-5.
echo.
pause
cls
goto start_backend_only

:exit_script
color 07
exit /b 0

:end
endlocal