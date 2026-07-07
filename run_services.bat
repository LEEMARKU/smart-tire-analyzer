@echo off
REM ============================================================================
REM Smart Tire Analyzer — AUTO DEPLOY
REM Double-click this script to automatically build & deploy the full stack:
REM   1. Local Python development (train + server without Docker)
REM   2. Docker Desktop (docker-compose: backend + frontend + redis + nginx)
REM   3. Kubernetes     (kubectl apply: backend + frontend deployments)
REM   4. Health check   (verify all services respond)
REM ============================================================================

setlocal enabledelayedexpansion

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

set "COMPOSE_FILE=%REPO_ROOT%\deployment\docker\docker-compose.yml"
set "K8S_DIR=%REPO_ROOT%\deployment\kubernetes"
set "K8S_NS=smart-tire"

set "ALL_OK=1"
set "SKIP_K8S=0"

title Smart Tire Analyzer — Auto Deploy

:MENU
cls
echo.
echo  ============================================================
echo       SMART TIRE ANALYZER — LAUNCH MENU
echo  ============================================================
echo.
echo  1. LOCAL DEV MODE (no Docker required)
echo     - Analyze dataset and train model
echo     - Start backend API
echo     - Open dashboard
echo.
echo  2. FRONTEND DEV SERVER
echo     - Start Next.js frontend at http://localhost:3000
echo.
echo  3. DOCKER DEPLOY (Docker Desktop required)
echo     - Build and start all containers
echo     - backend + frontend + redis + nginx
echo.
echo  4. EXIT
echo.
echo  ------------------------------------------------------------
echo.
set /p "CHOICE=Select option (1-4): "
if "%CHOICE%"=="1" goto LOCAL_DEV
if "%CHOICE%"=="2" goto FRONTEND_DEV
if "%CHOICE%"=="3" goto DOCKER_DEPLOY
if "%CHOICE%"=="4" exit /b 0
goto MENU

REM ============================================================================
REM LOCAL DEV MODE
REM ============================================================================
:LOCAL_DEV
cls
echo.
echo  ============================================================
echo       LOCAL DEV MODE
echo  ============================================================
echo.
echo  This will guide you through:
echo    1. Verify Python environment
echo    2. Analyze dataset and train model
echo    3. Start backend API server
echo.
echo  ------------------------------------------------------------
echo.

REM Check Python
python --version >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [FAIL] Python not found. Install Python 3.10+ first.
    pause
    goto MENU
)
echo  [OK]   Python found.

REM Check .env
if not exist "%REPO_ROOT%\.env" (
    if exist "%REPO_ROOT%\.env.example" (
        copy /Y "%REPO_ROOT%\.env.example" "%REPO_ROOT%\.env" >nul
        echo  [OK]   Created .env from .env.example.
    )
)

echo.
echo  [Step 1] Analyze dataset and train model?
echo  This will auto-select the best architecture for your dataset.
echo.
set /p "TRAIN=Run smart training? (Y/N): "
if /i "!TRAIN!"=="Y" (
    echo.
    echo  Starting smart training...
    echo  (This may take 30-60 minutes)
    echo.
    cd /d "%REPO_ROOT%"
    python scripts\train_smart.py
    if !ERRORLEVEL! NEQ 0 (
        echo  [FAIL] Training failed.
        pause
        goto MENU
    )
    echo  [OK]   Training complete.
) else (
    echo  Skipping training.
    if not exist "%REPO_ROOT%\ai_model\saved_models\hybrid_torch\model_best.pt" (
        echo  [WARN] No trained model found. Run training first.
    )
)

echo.
echo  [Step 2] Start backend API server?
echo.
set /p "START=Start server? (Y/N): "
if /i "!START!"=="Y" (
    echo.
    echo  Starting backend on http://localhost:8000
    echo  API docs at http://localhost:8000/docs
    echo.
    start "Smart Tire Backend" cmd /c "cd /d "%REPO_ROOT%\backend" && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
    echo  [OK]   Backend started in a new window.
    timeout /t 3 /nobreak >nul
    start http://localhost:8000/docs
)

echo.
echo  ------------------------------------------------------------
echo   LOCAL DEV MODE COMPLETE
echo  ------------------------------------------------------------
echo.
echo  Backend:  http://localhost:8000
echo  API Docs: http://localhost:8000/docs
echo  Frontend: http://localhost:3000 (run option 2 to start)
echo.
echo  New Features:
echo  - Live Chat:      /live-chat  (tire-only Llama 3.3 AI assistant)
echo  - Voice AI:       /technical-support (OmniDimension voice agent)
echo  - Contact:        /contact
echo.
echo  To analyze a tire image:
echo    python scripts\infer.py --image path\to\image.jpg
echo.
pause
goto MENU

REM ============================================================================
REM FRONTEND DEV SERVER
REM ============================================================================
:FRONTEND_DEV
cls
echo.
echo  ============================================================
echo       FRONTEND DEV SERVER
echo  ============================================================
echo.
echo  Starting Next.js frontend on http://localhost:3000
echo.

REM Check Node.js
where node >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [FAIL] Node.js not found. Install Node.js 18+ first.
    pause
    goto MENU
)
echo  [OK]   Node.js found.

REM Check if node_modules exists
if not exist "%REPO_ROOT%\frontend\node_modules" (
    echo  [INFO] Installing frontend dependencies...
    cd /d "%REPO_ROOT%\frontend"
    npm install
    if !ERRORLEVEL! NEQ 0 (
        echo  [FAIL] npm install failed.
        pause
        goto MENU
    )
    echo  [OK]   Dependencies installed.
)

cd /d "%REPO_ROOT%\frontend"
echo  Starting Next.js dev server...
start "Smart Tire Frontend" cmd /c "cd /d "%REPO_ROOT%\frontend" && npx next dev -p 3000"
echo  [OK]   Frontend started in a new window.
timeout /t 3 /nobreak >nul
start http://localhost:3000
echo.
echo  Frontend URL:     http://localhost:3000
echo  Live Chat:        http://localhost:3000/live-chat
echo  Technical Support: http://localhost:3000/technical-support
echo  Contact:          http://localhost:3000/contact
echo.
pause
goto MENU

REM ============================================================================
REM DOCKER DEPLOY MODE
REM ============================================================================
:DOCKER_DEPLOY
cls
echo.
echo  ============================================================
echo       SMART TIRE ANALYZER — DOCKER DEPLOY
echo  ============================================================
echo.
echo  This script will automatically:
echo    1. Check prerequisites (Docker Desktop, kubectl)
echo    2. Build Docker images for backend ^& frontend
echo    3. Deploy full stack via Docker Compose
echo    4. Deploy to Kubernetes (optional)
echo    5. Verify all services are healthy
echo.
echo  Press Ctrl+C at any time to abort.
echo  ------------------------------------------------------------
echo.
pause

REM ============================================================================
REM PHASE 1 — PREREQUISITES
REM ============================================================================
cls
echo.
echo  ============================================================
echo     [Phase 1/5] — Checking Prerequisites
echo  ============================================================
echo.

echo  [1/3] Checking Docker Desktop...
docker info >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [FAIL] Docker Desktop is NOT running.
    echo         Start Docker Desktop and try again.
    set "ALL_OK=0"
    goto RESULT
)
echo  [OK]   Docker Desktop is running.
echo.

echo  [2/3] Checking kubectl and Kubernetes cluster...
kubectl version --client >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [WARN] kubectl not found. K8s deployment will be skipped.
    set "SKIP_K8S=1"
) else (
    kubectl cluster-info >nul 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo  [WARN] Cannot connect to K8s cluster. K8s deployment will be skipped.
        set "SKIP_K8S=1"
    ) else (
        echo  [OK]   kubectl found and connected to K8s cluster.
    )
)
echo.

echo  [3/3] Checking environment file...
if not exist "%REPO_ROOT%\.env" (
    if exist "%REPO_ROOT%\.env.example" (
        copy /Y "%REPO_ROOT%\.env.example" "%REPO_ROOT%\.env" >nul
        echo  [OK]   Created .env from .env.example.
    ) else (
        echo  [WARN] No .env found. Using defaults.
    )
) else (
    echo  [OK]   .env file exists.
)
echo.
pause

REM ============================================================================
REM PHASE 2 — BUILD DOCKER IMAGES
REM ============================================================================
cls
echo.
echo  ============================================================
echo     [Phase 2/5] — Building Docker Images
echo  ============================================================
echo.
echo  This may take 5-15 minutes on first run...
echo.
cd /d "%REPO_ROOT%"
docker compose -f "%COMPOSE_FILE%" build 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo  [FAIL] Docker image build failed. Check Docker Desktop logs.
    set "ALL_OK=0"
    goto RESULT
)
echo.
echo  [OK]   Docker images built successfully.
echo.
pause

REM ============================================================================
REM PHASE 3 — DEPLOY VIA DOCKER COMPOSE
REM ============================================================================
cls
echo.
echo  ============================================================
echo     [Phase 3/5] — Deploying via Docker Compose
echo  ============================================================
echo.

echo  Starting full container stack in detached mode...
docker compose -f "%COMPOSE_FILE%" up -d 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo  [FAIL] Docker Compose deploy failed.
    set "ALL_OK=0"
    goto RESULT
)
echo.
echo  [OK]   Docker Compose stack started.
echo.
echo  Waiting for backend to become healthy (up to 3 minutes)...
echo.

set "BACKEND_HEALTHY=0"
for /l %%i in (1,1,36) do (
    curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > "%TEMP%\sta_health.tmp" 2>nul
    set /p HEALTH=<"%TEMP%\sta_health.tmp"
    if "!HEALTH!"=="200" (
        set "BACKEND_HEALTHY=1"
        echo.
        echo  [OK]   Backend is healthy (HTTP 200).
        goto COMPOSE_DONE
    )
    echo     Attempt %%i/36 — waiting 5s...
    timeout /t 5 /nobreak >nul
)

:COMPOSE_DONE
if "!BACKEND_HEALTHY!"=="0" (
    echo.
    echo  [WARN] Backend health check timed out (3 min).
    echo         It may still be loading models. Check logs:
    echo           docker logs smart-tire-backend
)
echo.
echo  [OK]   Docker Compose deployment complete.
echo         Backend:  http://localhost:8000  (API docs: /docs)
echo         Frontend: http://localhost:8081
echo         Nginx:    http://localhost:8080
echo         Redis:    localhost:6379
echo.
pause

REM ============================================================================
REM PHASE 4 — DEPLOY TO KUBERNETES
REM ============================================================================
cls
if "%SKIP_K8S%"=="1" (
    echo.
    echo  ============================================================
    echo     [Phase 4/5] — Kubernetes  [SKIPPED]
    echo  ============================================================
    echo.
    echo  Skipping K8s deployment because kubectl/K8s cluster
    echo  is not available.
    echo.
    pause
    goto HEALTH_CHECK
)

echo.
echo  ============================================================
echo     [Phase 4/5] — Deploying to Kubernetes
echo  ============================================================
echo.

echo  [1/6] Creating namespace "%K8S_NS%"...
kubectl create namespace %K8S_NS% --dry-run=client -o yaml 2>nul | kubectl apply -f - >nul
echo  [OK]   Namespace ready.
echo.

echo  [2/6] Applying backend deployment...
kubectl apply -f "%K8S_DIR%\deployment.yaml" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [FAIL] Backend deployment failed.
    set "ALL_OK=0"
) else (
    echo  [OK]   Backend deployment applied.
)
echo.

echo  [3/6] Applying frontend deployment...
if exist "%K8S_DIR%\frontend-deployment.yaml" (
    kubectl apply -f "%K8S_DIR%\frontend-deployment.yaml" 2>&1
    if !ERRORLEVEL! NEQ 0 (
        echo  [WARN] Frontend deployment apply had issues.
    ) else (
        echo  [OK]   Frontend deployment applied.
    )
) else (
    echo  [WARN] frontend-deployment.yaml not found. Skipping.
)
echo.

echo  [4/6] Applying backend service...
kubectl apply -f "%K8S_DIR%\service.yaml" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [WARN] Service apply had issues.
) else (
    echo  [OK]   Backend service applied.
)
echo.

echo  [5/6] Applying HPA (optional)...
kubectl apply -f "%K8S_DIR%\hpa.yaml" 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo  [WARN] HPA apply failed (optional).
) else (
    echo  [OK]   HPA applied.
)
echo.

echo  [6/6] Waiting for rollout (up to 5 min)...
echo.
kubectl -n %K8S_NS% rollout status deployment/smart-tire-backend --timeout=300s
if !ERRORLEVEL! NEQ 0 (
    echo  [WARN] Backend rollout timed out.
)
echo.

if exist "%K8S_DIR%\frontend-deployment.yaml" (
    kubectl -n %K8S_NS% rollout status deployment/smart-tire-frontend --timeout=120s 2>nul
)

echo.
echo  [OK]   Kubernetes deployment complete.
echo.
echo  K8s Pods:
kubectl -n %K8S_NS% get pods -o wide 2>nul
echo.
echo  K8s Services:
kubectl -n %K8S_NS% get svc -o wide 2>nul
echo.
pause

REM ============================================================================
REM PHASE 5 — HEALTH CHECK DASHBOARD
REM ============================================================================
:HEALTH_CHECK
cls
echo.
echo  ============================================================
echo     [Phase 5/5] — Service Health Dashboard
echo  ============================================================
echo.
echo  Checking all services...
echo.

echo  ┌─────────────────────────────────────────────────────────────────────┐
echo  │ SERVICE                  STATUS       URL                          │
echo  ├─────────────────────────────────────────────────────────────────────┤

set "S1=—"
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health > "%TEMP%\sta1.tmp" 2>nul
set /p S1=<"%TEMP%\sta1.tmp"
if "!S1!"=="200" (echo  │ Backend (Compose)        HEALTHY     http://localhost:8000) else (echo  │ Backend (Compose)        DOWN        http://localhost:8000)

set "S2=—"
curl -s -o nul -w "%%{http_code}" http://localhost:8081 > "%TEMP%\sta2.tmp" 2>nul
set /p S2=<"%TEMP%\sta2.tmp"
if "!S2!"=="200" (echo  │ Frontend (Compose)       HEALTHY     http://localhost:8081) else (echo  │ Frontend (Compose)       DOWN        http://localhost:8081)

set "S3=—"
curl -s -o nul -w "%%{http_code}" http://localhost:8080 > "%TEMP%\sta3.tmp" 2>nul
set /p S3=<"%TEMP%\sta3.tmp"
if "!S3!"=="200" (echo  │ Nginx Proxy              HEALTHY     http://localhost:8080) else (echo  │ Nginx Proxy              DOWN        http://localhost:8080)

set "S4=—"
curl -s -o nul -w "%%{http_code}" http://localhost:8000/docs > "%TEMP%\sta4.tmp" 2>nul
set /p S4=<"%TEMP%\sta4.tmp"
if "!S4!"=="200" (echo  │ API Docs (Swagger)       HEALTHY     http://localhost:8000/docs) else (echo  │ API Docs (Swagger)       DOWN        http://localhost:8000/docs)

if "%SKIP_K8S%"=="0" (
    kubectl -n %K8S_NS% get pods -o wide 2>nul | findstr /R "backend.*Running" >nul
    if !ERRORLEVEL! EQU 0 (echo  │ K8s Backend Pod          HEALTHY     ) else (echo  │ K8s Backend Pod          PENDING     )

    if exist "%K8S_DIR%\frontend-deployment.yaml" (
        kubectl -n %K8S_NS% get pods -o wide 2>nul | findstr /R "frontend.*Running" >nul
        if !ERRORLEVEL! EQU 0 (echo  │ K8s Frontend Pod         HEALTHY     ) else (echo  │ K8s Frontend Pod         PENDING     )
    )
)

echo  └─────────────────────────────────────────────────────────────────────┘

del "%TEMP%\sta*.tmp" 2>nul
echo.

REM ============================================================================
REM DEPLOYMENT SUMMARY
REM ============================================================================
:RESULT
echo.
echo  ============================================================
echo                    DEPLOYMENT SUMMARY
echo  ============================================================
echo.
echo  Docker Compose (Docker Desktop):
echo    Backend API:    http://localhost:8000
echo    API Docs:       http://localhost:8000/docs
echo    Frontend App:   http://localhost:8081
echo    Nginx Proxy:    http://localhost:8080
echo    Redis:          localhost:6379
echo.
echo  To view container logs:
echo    docker compose -f deployment\docker\docker-compose.yml logs -f
echo.

if "%SKIP_K8S%"=="0" (
    echo  Kubernetes (namespace: %K8S_NS%):
    echo    View pods:     kubectl -n %K8S_NS% get pods
    echo    View logs:     kubectl -n %K8S_NS% logs deployment/smart-tire-backend
    echo    Port-forward:  kubectl -n %K8S_NS% port-forward svc/smart-tire-backend-svc 8000:80
    echo.
)

if "!ALL_OK!"=="0" (
    echo  [!] WARNING: Some services failed to deploy.
    echo      Check Docker Desktop logs and try again.
    echo.
) else (
    echo  [OK] All services deployed successfully!
    echo.
)

echo  ------------------------------------------------------------
echo   Open Frontend in browser:  http://localhost:8081
echo   Open API Docs:             http://localhost:8000/docs
echo  ------------------------------------------------------------
echo.

:ASK_OPEN
set /p "OPEN=Open Frontend in browser? (Y/N): "
if /i "!OPEN!"=="Y" (
    start http://localhost:8081
    echo  [OK] Browser opened.
) else if /i "!OPEN!"=="N" (
    echo  Skipped.
) else (
    goto ASK_OPEN
)
echo.

:ASK_STOP
echo  ------------------------------------------------------------
echo   The stack is still running in the background.
echo   Would you like to stop all services?
echo  ------------------------------------------------------------
echo.
set /p "STOP=Stop services? (Y/N): "
if /i "!STOP!"=="Y" goto SHUTDOWN
if /i "!STOP!"=="N" (
    echo.
    echo  Services will keep running. Close this window to exit.
    echo  To stop later, re-run this script or use:
    echo    docker compose -f deployment\docker\docker-compose.yml down
    echo.
    timeout /t 5 /nobreak >nul
    exit /b 0
)
goto ASK_STOP

REM ============================================================================
REM SHUTDOWN
REM ============================================================================
:SHUTDOWN
cls
echo.
echo  ============================================================
echo                    SHUTTING DOWN
echo  ============================================================
echo.

echo  [1/2] Stopping Docker Compose stack...
docker compose -f "%COMPOSE_FILE%" down 2>nul
if !ERRORLEVEL! EQU 0 (
    echo  [OK]   Docker stack stopped.
) else (
    echo  [WARN] Docker Compose down failed.
)
echo.

if "%SKIP_K8S%"=="0" (
    echo  [2/2] Deleting Kubernetes resources...
    kubectl delete namespace %K8S_NS% --ignore-not-found 2>nul
    echo  [OK]   K8s resources deleted.
)
echo.
echo  All services stopped. Goodbye!
echo.
timeout /t 3 /nobreak >nul
exit /b 0
