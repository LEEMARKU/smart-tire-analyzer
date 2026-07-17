# Smart Tire Analyzer — Deployment & Operations Guide

## Local Development Setup

### Prerequisites
- Python 3.10+ (for backend)
- Node.js 18+ (for frontend)
- Git

### Quick Start (Windows)
```bash
cd smart-tire-analyzer
run_services.bat
# Choose option [3] to start both backend and frontend
```

### Quick Start (macOS/Linux)
```bash
# Backend
cd backend
export PYTHONPATH=backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (in new terminal)
cd frontend
npm install
npm run dev
```

### Verify Installation
```bash
# Backend health check
curl http://localhost:8000/health
# Expected: {"status": "online"}

# Frontend
open http://localhost:3000
```

---

## Docker Deployment

### Build Locally
```bash
docker compose -f deployment/docker/docker-compose.yml build
```

### Run Locally
```bash
docker compose -f deployment/docker/docker-compose.yml up

# Services available at:
# - Backend API: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - Frontend: http://localhost:3000
# - Nginx (prod): http://localhost:80
```

### Push to Registry
```bash
# Tag images
docker tag smart-tire-analyzer-backend:latest <registry>/smart-tire-analyzer-backend:1.0.0
docker tag smart-tire-analyzer-frontend:latest <registry>/smart-tire-analyzer-frontend:1.0.0

# Push
docker push <registry>/smart-tire-analyzer-backend:1.0.0
docker push <registry>/smart-tire-analyzer-frontend:1.0.0
```

---

## Kubernetes Deployment

### Prerequisites
- kubectl 1.20+
- Kubernetes cluster (minikube, EKS, GKE, AKS, etc.)
- Docker images in a registry

### Create Namespace & Secrets
```bash
# Create namespace
kubectl create namespace smart-tire

# Create secrets for API keys (⚠️ replace with real values)
kubectl create secret generic smart-tire-backend-secrets \
  --namespace smart-tire \
  --from-literal=GEMINI_API_KEYS="your-gemini-key" \
  --from-literal=GOOGLE_MAPS_API_KEYS="your-maps-key" \
  --from-literal=OPENWEATHER_API_KEYS="your-weather-key" \
  --from-literal=MAPILLARY_API_KEYS="your-mapillary-key" \
  --from-literal=JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')" \
  --from-literal=NEXTAUTH_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

### Update deployment.yaml
```yaml
# Change image reference to your registry
image: <registry>/smart-tire-analyzer-backend:1.0.0
```

### Apply Manifests
```bash
# Dry-run (validate)
kubectl apply -f deployment/kubernetes/ --namespace smart-tire --dry-run=client

# Apply
kubectl apply -f deployment/kubernetes/ --namespace smart-tire
```

### Verify Deployment
```bash
# Check pods
kubectl get pods -n smart-tire

# Check services
kubectl get svc -n smart-tire

# Port-forward to test
kubectl port-forward -n smart-tire svc/smart-tire-backend-svc 8000:80

# Test health
curl http://localhost:8000/health
```

### Scale Deployment
```bash
# Manual scaling
kubectl scale deployment smart-tire-backend -n smart-tire --replicas=3

# Or let HPA handle it (configured for 2-4 replicas)
kubectl get hpa -n smart-tire
```

---

## Environment Configuration

### Local Development (.env)
```env
# ─── API Keys ───
GEMINI_API_KEYS=your-dev-key-1,your-dev-key-2
GOOGLE_MAPS_API_KEYS=your-dev-key
OPENWEATHER_API_KEYS=your-dev-key
MAPILLARY_API_KEYS=your-dev-key

# ─── Database ───
DATABASE_URL=sqlite+aiosqlite:///./data/smart_tire.db

# ─── Auth ───
AUTH_ENABLED=false
JWT_SECRET=dev-secret-only-32-chars-min!

# ─── CORS ───
BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8081

# ─── Redis ───
REDIS_URL=redis://localhost:6379/0

# ─── Frontend ───
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Production (.env.production)
```env
# ─── API Keys (rotate regularly) ───
GEMINI_API_KEYS=prod-key-1,prod-key-2,prod-key-3
GOOGLE_MAPS_API_KEYS=prod-key-1,prod-key-2
OPENWEATHER_API_KEYS=prod-key-1,prod-key-2
MAPILLARY_API_KEYS=prod-key-1,prod-key-2

# ─── Database (PostgreSQL recommended) ───
DATABASE_URL=postgresql://user:password@prod-db.example.com:5432/smart_tire

# ─── Auth (MUST enable) ───
AUTH_ENABLED=true
JWT_SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(48))')
API_KEY=<generate-secure-key>

# ─── CORS ───
BACKEND_CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# ─── Redis (managed service recommended) ───
REDIS_URL=redis://redis-prod.example.com:6379/0

# ─── Frontend ───
NEXT_PUBLIC_API_BASE_URL=https://api.your-domain.com

# ─── Features ───
NOTIFICATIONS_ENABLED=true
AUTO_RETRAIN=true
```

---

## Database Migrations

### Check Migration Status
```bash
python scripts/migrate_db.py
```

### Apply Migrations
```bash
python scripts/migrate_db.py --apply
```

### Rollback Last Migration
```bash
python scripts/migrate_db.py --rollback
```

### Migrate to PostgreSQL
```bash
# 1. Export SQLite data
sqlite3 data/smart_tire.db ".dump" > dump.sql

# 2. Create PostgreSQL database
psql -U postgres -d smart_tire < dump.sql

# 3. Update DATABASE_URL
DATABASE_URL=postgresql://user:password@localhost:5432/smart_tire

# 4. Run migrations
python scripts/migrate_db.py --apply
```

---

## Monitoring & Logging

### Prometheus Metrics
```bash
# Collect metrics
curl http://localhost:8000/metrics

# Key metrics:
# - smart_tire_analyze_requests_total
# - smart_tire_analyze_latency_seconds
# - smart_tire_model_ready
```

### Structured Logging
```python
# All logs go to stdout in JSON format (Docker/K8s compatible)
# Example: {"timestamp": "...", "level": "INFO", "logger": "smart_tire_api", "message": "..."}
```

### View Logs
```bash
# Docker
docker compose logs -f backend

# Kubernetes
kubectl logs -f deployment/smart-tire-backend -n smart-tire
```

---

## Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.10+

# Check dependencies
pip install -r backend/requirements.txt

# Check PYTHONPATH
export PYTHONPATH=backend

# Verify database
python scripts/migrate_db.py
```

### Frontend won't build
```bash
# Clear cache
rm -rf frontend/.next frontend/node_modules

# Reinstall
cd frontend
npm install

# Rebuild
npm run build
```

### Docker build fails
```bash
# Check Dockerfile paths
ls -la deployment/docker/

# Rebuild with no cache
docker compose build --no-cache

# View build output
docker compose build --verbose
```

### Kubernetes pod won't start
```bash
# Check pod status
kubectl describe pod <pod-name> -n smart-tire

# Check resource limits
kubectl top pod -n smart-tire

# Check logs
kubectl logs <pod-name> -n smart-tire --tail=100

# Check events
kubectl get events -n smart-tire
```

---

## Performance Tuning

### Backend
```python
# Increase worker count in uvicorn (deployment.yaml)
command: ["uvicorn", "app.main:app", "--workers", "4", "--port", "8000"]

# Increase memory/CPU limits
resources:
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

### Frontend
```bash
# Enable static generation
# Use ISR (Incremental Static Regeneration) for pages that change infrequently

# Optimize images
# Use next/image for automatic optimization
```

### Database
```bash
# Enable WAL (Write-Ahead Logging) for SQLite (dev only)
sqlite3 data/smart_tire.db "PRAGMA journal_mode=WAL;"

# For production, migrate to PostgreSQL and add indexes
CREATE INDEX idx_analysis_session_id ON analysis_results(session_id);
CREATE INDEX idx_feedback_session_id ON feedback_records(session_id);
```

---

## Backup & Recovery

### Backup Database
```bash
# SQLite
cp data/smart_tire.db data/smart_tire_backup_$(date +%Y%m%d).db

# PostgreSQL
pg_dump smart_tire > smart_tire_backup_$(date +%Y%m%d).sql
```

### Backup Model Checkpoints
```bash
tar -czf ai_model_backup_$(date +%Y%m%d).tar.gz ai_model/saved_models/
```

### Restore from Backup
```bash
# SQLite
cp data/smart_tire_backup_20260717.db data/smart_tire.db

# PostgreSQL
psql smart_tire < smart_tire_backup_20260717.sql

# Models
tar -xzf ai_model_backup_20260717.tar.gz
```

---

## Security Checklist

- [ ] Change all default API keys (Gemini, Maps, Weather, Mapillary)
- [ ] Set strong JWT_SECRET (use: `python -c 'import secrets; print(secrets.token_urlsafe(48))'`)
- [ ] Enable AUTH_ENABLED=true in production
- [ ] Use HTTPS/TLS (Ingress/LoadBalancer)
- [ ] Set up network policies (K8s NetworkPolicy)
- [ ] Enable pod security standards (K8s Pod Security Standards)
- [ ] Rotate API keys regularly (monthly recommended)
- [ ] Monitor logs for suspicious activity
- [ ] Set up backup & disaster recovery
- [ ] Document runbook for incident response

---

## Useful Commands

```bash
# Local testing
run_services.bat            # Windows launcher
npm --prefix frontend run build  # Build frontend
pytest tests/ -v             # Run backend tests

# Docker
docker compose logs -f       # Stream logs
docker compose down -v       # Stop & remove volumes

# Kubernetes
kubectl get all -n smart-tire    # View all resources
kubectl delete -f deployment/kubernetes/ -n smart-tire  # Cleanup

# Database
python scripts/migrate_db.py      # Check migrations
sqlite3 data/smart_tire.db        # Connect to DB
```

---

## Support & Troubleshooting

For issues:
1. Check logs: `docker compose logs` or `kubectl logs`
2. Review FIX_PLAN.md for known issues
3. Check AUDIT_SUMMARY.md for security/code quality notes
4. See ARCHITECTURE.md for design decisions
