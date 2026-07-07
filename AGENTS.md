# AGENTS.md - Smart Tire Analyzer

## Training New Models
```bash
# Full training (auto-configures based on dataset size)
python train_new_models.py

# Quick test (2 epochs per stage)
python train_new_models.py --epochs 2 --skip-tflite

# Dry-run (analysis only)
python train_new_models.py --dry-run
```

## Backend Server
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Frontend Dev Server
```bash
cd frontend
npm install    # first time only
npm run dev    # starts at http://localhost:3000
```

## Live Chat (Llama 3.3 tire-only AI)
The live chat API uses a strict tire-only system prompt.
To use with Ollama locally:
```bash
ollama pull llama3.3:latest
# Set LIVE_CHAT_PROVIDER=llama3 in .env
```

## Voice AI Support (OmniDimension)
Voice AI agent for tire & technical support on `/technical-support`.
```bash
# Get API key at https://omnidim.io
# Set OMNIDIM_API_KEY=your_key in .env

# Install
pip install omnidimension
```

## Android App Compilation
```bash
cd android-app
./gradlew assembleDebug
```

## Lint & Typecheck
```bash
# Python
ruff check ai_model/ backend/ android-app/models/
mypy ai_model/ backend/ android-app/models/

# Android (from android-app directory)
./gradlew lint
```
