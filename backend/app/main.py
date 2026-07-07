"""
FastAPI Backend — Smart Tire Analyzer Main Application Entry Point
"""

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.routes import inference, feedback, health, history, enterprise, metrics, registry, auth, ws, video, omnidim
from app.database.db import ensure_database_ready
from app.services.cache_service import get_cache, init_cache
from app.services.inference_service import InferenceService
from app.services.api_key_rotator import initialize_rotators, get_gemini_rotator, get_weather_rotator, get_maps_rotator, get_mapillary_rotator
from app.services.security_service import SecurityService
from app.config import settings
from api_integrations.gemini.gemini_client import get_gemini_client
from api_integrations.weather.weather_client import get_weather_client
from api_integrations.google_maps.maps_client import get_maps_client

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("smart_tire_api")

# ─── Global model service ───────────────────────────────────────────────────
inference_service: InferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global inference_service
    logger.info("🚀 Starting Smart Tire Analyzer API...")
    settings.log_startup_config()

    # ─── Initialize API Key Rotators ────────────────────────────────────────
    logger.info("Initializing API key rotators...")
    initialize_rotators(
        gemini_keys=settings.get_gemini_keys(),
        weather_keys=settings.get_weather_keys(),
        maps_keys=settings.get_maps_keys(),
        mapillary_keys=settings.get_mapillary_keys(),
    )
    
    # Update API clients with rotators
    gemini_rotator = get_gemini_rotator()
    weather_rotator = get_weather_rotator()
    maps_rotator = get_maps_rotator()
    
    get_gemini_client(rotator=gemini_rotator)
    get_weather_client(rotator=weather_rotator)
    get_maps_client(rotator=maps_rotator)
    logger.info("✅ API key rotators initialized and connected to clients")

    # Create DB tables
    await ensure_database_ready(force=True)
    logger.info("✅ Database tables ready")

    # Initialize Redis cache
    await init_cache(redis_url=settings.REDIS_URL)
    logger.info("✅ Cache service initialized")

    # Load AI model once at startup
    inference_service = InferenceService()
    await inference_service.initialize()
    app.state.inference_service = inference_service
    if inference_service.is_ready():
        logger.info("AI model loaded and warm")
    else:
        logger.warning("AI model is not ready: %s", inference_service._load_error)

    yield  # App is running

    logger.info("🛑 Shutting down Smart Tire Analyzer API...")
    cache_svc = get_cache()
    await cache_svc.close()
    if inference_service is not None:
        await inference_service.cleanup()


# ─── App factory ────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Tire Analyzer API",
        description=(
            "Production AI-powered tire intelligence system. "
            "Predicts tread depth, wear patterns, health score, and remaining life "
            "using CNN + ViT + RNN + ANN hybrid model with Gemini AI reasoning."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ─── Rate limiter (in-memory, per IP) ─────────────────────────────────────
    rate_limit_buckets: dict[str, list[float]] = defaultdict(list)
    RATE_LIMIT_MAX = 60
    RATE_LIMIT_WINDOW = 60.0

    @app.middleware("http")
    async def rate_limiter(request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = now - RATE_LIMIT_WINDOW
        buckets = rate_limit_buckets[client_ip]
        while buckets and buckets[0] < window:
            buckets.pop(0)
        if len(buckets) >= RATE_LIMIT_MAX:
            return JSONResponse(status_code=429, content={"error": "Too many requests"})
        buckets.append(now)
        return await call_next(request)

    # ─── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ─── Request timing middleware ───────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration:.2f}"
        return response

    @app.middleware("http")
    async def optional_auth_guard(request: Request, call_next):
        if not settings.AUTH_ENABLED:
            return await call_next(request)

        public_prefixes = (
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/enterprise/security/demo-token",
        )
        path = request.url.path
        if request.method == "OPTIONS" or path in public_prefixes or path.startswith("/health") or path.startswith("/auth"):
            return await call_next(request)

        security = SecurityService()
        api_key = request.headers.get("x-api-key")
        if settings.API_KEY and api_key and security.verify_api_key(api_key):
            request.state.user = {"sub": "api-key-client", "role": "technician", "auth": "api_key"}
            return await call_next(request)

        verification = security.verify_authorization_header(
            request.headers.get("authorization")
        )
        if not verification.get("valid"):
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "detail": verification.get("reason")},
            )
        request.state.user = verification.get("claims", {})
        return await call_next(request)

    # ─── CSRF Protection (double-submit cookie pattern) ───────────────────────
    # CSRF is only enforced when the frontend explicitly sends X-CSRF-Token headers.
    # The middleware sets a signed CSRF cookie on every response so the frontend
    # CAN opt in by including the X-CSRF-Token header on state-changing requests.
    #
    # This is a progressive enhancement: the API works without CSRF protection for
    # development, but frontend clients can add CSRF by reading the cookie value
    # and sending it back. The primary CSRF defense is the CORS origin restriction
    # and Bearer token auth (not cookie-based sessions for state changes).
    CSRF_SECRET = settings.JWT_SECRET or "csrf-dev-fallback-32-char-minimum!!"
    CSRF_COOKIE_NAME = "sta-csrf-token"
    SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next):
        response = await call_next(request)

        # Only set CSRF cookie on state-changing responses
        if request.method not in SAFE_METHODS:
            csrf_token = str(uuid.uuid4())
            sign = hmac.new(
                CSRF_SECRET.encode("utf-8"),
                csrf_token.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:16]
            signed_token = f"{csrf_token}.{sign}"
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=signed_token,
                httponly=True,
                samesite="strict",
                secure=request.url.scheme == "https",
                max_age=86400,
            )

            # Validate if frontend sent CSRF header
            header_token = request.headers.get("X-CSRF-Token", "")
            existing_cookie = request.cookies.get(CSRF_COOKIE_NAME, "")
            if header_token and existing_cookie:
                try:
                    cookie_token, cookie_sign = existing_cookie.rsplit(".", 1)
                    expected_sign = hmac.new(
                        CSRF_SECRET.encode("utf-8"),
                        cookie_token.encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()[:16]
                    if not hmac.compare_digest(cookie_sign, expected_sign):
                        return JSONResponse(
                            status_code=403,
                            content={"error": "Invalid CSRF token"},
                        )
                    if header_token != cookie_token:
                        return JSONResponse(
                            status_code=403,
                            content={"error": "CSRF token mismatch"},
                        )
                except (ValueError, IndexError):
                    return JSONResponse(
                        status_code=403,
                        content={"error": "Malformed CSRF token"},
                    )

        return response

    # ─── Global exception handler (safe: never leaks internal details) ────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled exception [%s] on %s %s: %s",
            request_id, request.method, request.url.path, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
            },
        )

    # ─── Routers ─────────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(inference.router, prefix="/analyze", tags=["Analysis"])
    app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
    app.include_router(history.router, prefix="/history", tags=["History"])
    app.include_router(enterprise.router, prefix="/enterprise", tags=["Enterprise AI"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Monitoring"])
    app.include_router(registry.router, prefix="/registry", tags=["Model Registry"])
    app.include_router(ws.router)
    app.include_router(video.router)
    app.include_router(omnidim.router)

    # ─── Root endpoint ────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "service": "Smart Tire Analyzer API",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs",
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
