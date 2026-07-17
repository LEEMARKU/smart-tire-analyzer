# Smart Tire Analyzer — Architecture & Project Structure

## Directory Organization

```
smart-tire-analyzer/
├── ai_model/                      # AI/ML models (59 architectures)
│   ├── models/                    # Model implementations (CNN, RNN, Transformer, ANN)
│   ├── hybrid_torch/              # PyTorch training pipeline
│   │   ├── trainer.py             # Multi-task training orchestrator
│   │   ├── model.py               # Unified hybrid model architecture
│   │   ├── dataset.py             # HybridTireDataset with tread sequences
│   │   └── constants.py           # Labels, hyperparameters, model configs
│   ├── model_selector.py          # Auto-architecture recommender (6 tiers)
│   └── saved_models/              # Trained checkpoints (not in Git)
│
├── backend/                       # FastAPI server
│   ├── app/
│   │   ├── main.py                # App factory, lifespan, middleware
│   │   ├── config.py              # Settings, environment parsing
│   │   ├── database/
│   │   │   └── db.py              # SQLAlchemy setup, models
│   │   ├── routes/
│   │   │   ├── inference.py       # POST /analyze, GET /health
│   │   │   ├── feedback.py        # POST /feedback (continuous learning)
│   │   │   ├── history.py         # GET /history, GET /registry
│   │   │   ├── enterprise.py      # Gemini AI reasoning, Maps, Weather
│   │   │   ├── auth.py            # JWT + API key auth
│   │   │   ├── metrics.py         # Prometheus metrics
│   │   │   ├── ws.py              # WebSocket endpoints
│   │   │   └── omnidim.py         # Voice AI (OmniDimension)
│   │   ├── services/
│   │   │   ├── inference_service.py      # Model loading & inference orchestration
│   │   │   ├── api_key_rotator.py        # Multi-key rotation for Gemini, Maps, Weather
│   │   │   ├── cache_service.py          # Redis caching
│   │   │   ├── security_service.py       # JWT verification, API key validation
│   │   │   └── continuous_learning_service.py # Auto-retraining on user corrections
│   │   └── models/
│   │       └── schemas.py         # Pydantic request/response schemas
│   └── requirements.txt           # Python dependencies (PyTorch, FastAPI, SQLAlchemy)
│
├── frontend/                      # Next.js 15 App Router frontend
│   ├── app/
│   │   ├── layout.tsx             # Root layout + providers
│   │   ├── page.tsx               # Home page
│   │   ├── live-chat/
│   │   │   ├── page.tsx           # Live Chat UI (Llama 3.3 AI)
│   │   │   └── api/               # Chat API route handlers
│   │   ├── technical-support/
│   │   │   ├── page.tsx           # Technical Support + Voice AI
│   │   │   └── api/               # Voice API handlers (OmniDimension)
│   │   ├── components/            # Shared UI components
│   │   ├── hooks/                 # Custom React hooks
│   │   └── lib/                   # Utilities (API client, etc.)
│   ├── next.config.mjs            # Next.js config (ESM)
│   ├── tsconfig.json              # TypeScript config (frontend-specific paths)
│   └── package.json               # Node dependencies
│
├── android-app/                   # Kotlin Android app (TFLite inference)
│   ├── app/src/main/
│   │   ├── assets/                # TFLite models (FP32, FP16, INT8)
│   │   ├── java/.../util/
│   │   │   └── TireInferenceEngine.kt  # TFLite runtime
│   │   └── java/.../viewmodel/
│   │       └── TireTwinViewModel.kt    # ViewModel + state management
│   └── build.gradle.kts           # Android build config
│
├── continuous_learning/           # ⚠️ LEGACY - Can be removed
│   ├── __init__.py                # (kept for import compatibility)
│   ├── feedback_service.py
│   ├── sample_storage.py
│   ├── model_versions/
│   ├── retraining/
│   └── wrong_predictions/
│
├── dataset/                       # Training data (not in Git)
│   ├── splits/
│   │   ├── train/labels.csv
│   │   ├── validation/labels.csv
│   │   └── test/labels.csv
│   ├── preprocessing/             # Image pipeline (rotation, shadow removal, GrabCut)
│   └── raw/
│
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.backend     # Multi-stage build for FastAPI
│   │   ├── Dockerfile.frontend    # Multi-stage build for Next.js
│   │   └── docker-compose.yml     # Local dev: backend + frontend + nginx + redis
│   └── kubernetes/
│       ├── deployment.yaml        # Backend Deployment (1 replica, HPA ready)
│       ├── service.yaml           # LoadBalancer Service + PVC
│       └── hpa.yaml               # HorizontalPodAutoscaler (2-4 replicas)
│
├── scripts/
│   ├── train_smart.py             # Auto-tier training entry point
│   ├── start_server.py            # Backend startup helper
│   ├── infer.py                   # Local inference CLI
│   ├── migrate_db.py              # Database migration tool
│   └── setup_env.py               # Environment setup
│
├── tests/                         # Pytest test suite
│   ├── test_inference_service_runtime.py
│   ├── test_infrastructure.py
│   ├── test_api_routes.py
│   └── conftest.py                # Fixtures
│
├── run_services.bat               # ✅ Multi-service launcher (Windows)
├── next.config.mjs                # ✅ Next.js ESM config (App Router)
├── tsconfig.json                  # ✅ Root TypeScript config (unified paths)
├── package.json                   # ✅ Root package.json (workspace root)
├── pyproject.toml                 # Python build config
├── pyrightconfig.json             # Python type checking
├── .env.example                   # Environment template
├── .env.production                # Production overrides
├── .gitignore                     # Git ignore rules (includes smart_tire.db)
├── Makefile                       # Build tasks
├── README.md                      # Project overview
├── FIX_PLAN.md                    # Known issues & solutions
├── AUDIT_SUMMARY.md               # Security & code quality audit
└── ARCHITECTURE.md                # This file
```

---

## Data Flow

### Inference Pipeline
```
1. User uploads tire image
   ↓
2. FastAPI /analyze endpoint
   ↓
3. [24-step preprocessing]
   - Auto-rotation, shadow removal, GrabCut, CLAHE, edge detection
   ↓
4. Hybrid model inference
   - CNN (EfficientNetV2-B0): local tread features (512-dim)
   - RNN (BiLSTM): sequential tread features (256-dim)
   - Fusion (Deep Dense ANN): combined representation (512-dim)
   ↓
5. Prediction heads
   - Tread depth (4 regression outputs)
   - Health score (0-10)
   - Remaining life (km)
   - Wear pattern (6-class)
   - Condition (3-class: safe/moderate/replace)
   ↓
6. Gemini AI reasoning layer
   - Context-aware advice, driving recommendations
   ↓
7. Enhanced with Maps/Weather APIs
   ↓
8. JSON response with full report
```

### Continuous Learning Loop
```
1. User submits feedback (correction)
   ↓
2. Stored in feedback_records table
   ↓
3. After N samples (≥10), trigger auto-retraining
   ↓
4. Fine-tune last 3 CNN blocks + all prediction heads
   ↓
5. Validate on holdout set
   ↓
6. If ↑ metrics, save new checkpoint
   ↓
7. Inference service loads updated model
```

---

## Auto-Model Selection (6 Tiers)

| Dataset Size | Tier | CNN | RNN | Fusion | Notes |
|---|---|---|---|---|---|
| 0–200 | **tiny** | ResNet18 | GRU | Standard FC | Mobile/edge |
| 200–500 | **very_small** | MobileNetV2 | BiGRU | MLP | Light deployment |
| 500–2000 | **small** | EfficientNetV2-B0 | BiLSTM | Deep Dense | Recommended for dev |
| 2000–5000 | **small_plus** | EfficientNetV2-B0 | Stacked LSTM | Self-Attention Fusion | Production baseline |
| 5000–15000 | **medium** | ConvNeXt | Encoder-Decoder LSTM | Cross-Modal Attention | Large enterprise |
| 15000+ | **large** | ConvNeXt | TCN | Multimodal Transformer Fusion | ViT optional | Advanced research |

---

## Database Schema

### Tables
- **analysis_results**: Session-level tire analysis records
- **users**: Technician accounts (optional, for enterprise)
- **feedback_records**: User corrections for continuous learning
- **_migrations**: Applied migration history

### Key Constraints
- `session_id` (UNIQUE) → one report per analysis session
- `feedback_records.session_id` (FK) → links to original analysis
- Supports SQLite (dev) and PostgreSQL (production)

---

## Deployment Targets

### Local Development
```bash
run_services.bat  # Windows launcher (validated Python & Node.js)
```

### Docker Compose (Local)
```bash
docker compose -f deployment/docker/docker-compose.yml up
# Runs: backend (port 8000) + frontend (port 3000) + nginx + redis
```

### Kubernetes (Production)
```bash
kubectl apply -f deployment/kubernetes/
# Creates: Deployment (1 replica) + Service + PVC + HPA (2-4 replicas)
```

---

## Dependencies & Versions

### Backend (Python 3.11)
- **FastAPI**: 0.111.0
- **PyTorch**: 2.4.1 (primary ML framework)
- **TensorFlow**: 2.16.1 (optional, for TFLite export)
- **SQLAlchemy**: 2.0.30
- **Pydantic**: 2.7.1

### Frontend (Node 20)
- **Next.js**: 15.0.0 (App Router)
- **React**: 19
- **TypeScript**: 5.7.3
- **Tailwind**: 4.2.0

### Mobile (Kotlin)
- **Android API 24+**
- **TensorFlow Lite** (on-device inference)

---

## Environment Variables

See `.env.example` for complete list. Key ones:
- `AUTH_ENABLED`: Enable JWT/API key auth
- `GEMINI_API_KEYS`: Comma-separated for rotation
- `DATABASE_URL`: SQLite (dev) or PostgreSQL (prod)
- `REDIS_URL`: Cache backend
- `NEXT_PUBLIC_API_BASE_URL`: Frontend→backend URL

---

## Security Notes

1. **API Key Rotation**: Built-in multi-key support for Gemini, Maps, Weather
2. **JWT Auth**: Configurable; defaults off for local dev
3. **Rate Limiting**: 60 req/min per IP
4. **CORS**: Hardened; only localhost in dev
5. **Database**: SQLite (dev), PostgreSQL (prod recommended)

---

## Testing Strategy

- **Backend**: pytest (8+ test files, CI/CD gated)
- **Frontend**: vitest (component tests) + Playwright (E2E)
- **Load**: k6/locust (API performance benchmarks)
- **Coverage**: Target 70%+ for critical paths

---

## Known Limitations & Future Work

1. **Continuous learning** requires manual retrain trigger (auto planned)
2. **On-device inference** (Android) uses older TFLite model (planned ViT.js update)
3. **Real-time video analysis** not yet implemented
4. **Federated learning** for multi-device training not yet implemented

---

## Maintenance

- **Database migrations**: `python scripts/migrate_db.py --apply`
- **Model checkpoints**: Auto-saved in `ai_model/saved_models/`
- **Logs**: Structured via `structlog`; sent to stdout (Docker/K8s compatible)
- **Metrics**: Prometheus format at `/metrics`
