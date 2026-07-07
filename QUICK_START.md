# Quick Start Guide

## 5-Minute Setup

### 1. Setup Environment

```bash
git clone <repo-url>
cd smart-tire-analyzer

# One-command: creates venv, installs deps, creates directories
python scripts/setup_env.py
```

### 2. Configure API Keys (Optional)

```bash
notepad .env       # Windows
nano .env          # Linux/macOS
```

Add your API keys if you want external context features:
```
GEMINI_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here
```

### 3. Place Dataset

Put tire images in `dataset/raw/tread_images/` organized by condition class:

```
dataset/raw/tread_images/
  safe/       # images of safe tires
  moderate/   # images of moderately worn tires
  replace/    # images of tires needing replacement
```

Then prepare labels:
```bash
python dataset/preprocessing/validate_images.py
```

### 4. Train Model (Auto-Config)

```bash
# Analyze dataset + train the best architecture automatically
python scripts/train_smart.py

# Or just see what it would recommend
python scripts/train_smart.py --analyze
```

The system:
- Counts your training samples
- Selects optimal CNN + RNN + Fusion from 59 architectures
- Trains in 2 stages: frozen encoder (30 epochs) then fine-tune (10 epochs)
- Saves best model to `ai_model/saved_models/hybrid_torch/model_best.pt`

### 5. Start Backend

```bash
python scripts/start_server.py
```

**API:** http://localhost:8000  
**Docs:** http://localhost:8000/docs

### 6. (Optional) Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.
Try **Live Chat** at `/live-chat` or **Technical Support** at `/technical-support`.

### 7. Test Inference

```bash
python scripts/infer.py --image path/to/tire.jpg
```

Or double-click `run_services.bat` and select **Local Dev Mode**.

---

## Alternative: One-Click Launch

Double-click `run_services.bat` and select:
- **1** — Local Dev Mode: train + start server interactively
- **2** — Docker Deploy: full stack via Docker Compose

---

## Running on Your Own Dataset

The system adapts to any dataset size:

| Samples | Tier | Typical Architecture |
|---------|------|-------------------|
| 0–200 | tiny | ResNet18 + GRU + Standard FC |
| 200–500 | very_small | MobileNetV2 + BiGRU + MLP |
| 500–2000 | small | EfficientNetV2-B0 + BiLSTM + Deep Dense Fusion |
| 2000–5000 | small_plus | EfficientNetV2-B0 + Stacked LSTM + Self-Attention Fusion |
| 5000–15000 | medium | ConvNeXt + Encoder-Decoder LSTM + Cross-Modal Attention |
| 15000+ | large | ConvNeXt + TCN + Multimodal Transformer Fusion |

---

## Preprocessing Pipeline (24 Steps)

All applied automatically during training and inference:

| Step | Technique | File |
|------|-----------|------|
| 1 | Auto rotation correction | `image_enhancement.py` |
| 2 | Shadow removal (LAB) | `image_enhancement.py` |
| 3 | Background removal (GrabCut) | `cnn/preprocessing.py` |
| 4 | Bilateral filtering | `image_enhancement.py` |
| 5 | Illumination correction | `image_enhancement.py` |
| 6 | CLAHE contrast enhancement | `cnn/preprocessing.py` |
| 7 | Sharpening | `cnn/preprocessing.py` |
| 8 | Edge detection (4th channel) | `cnn/preprocessing.py` |
| 9 | Resize (224x224) | `cnn/preprocessing.py` |
| 10 | ImageNet normalization | `cnn/preprocessing.py` |

---

## Troubleshooting

### "No module found"
```bash
.venv\Scripts\python scripts\setup_env.py
```

### Model not training
```bash
python scripts\train_smart.py --analyze
# Checks dataset size and recommends architecture
```

### Backend won't start
```bash
python -c "import torch; print(torch.__version__)"
# Ensure PyTorch is installed in your .venv
```

---

## Next Steps

- See full **architecture docs**: `docs/ARCHITECTURE.md`
- **Training guide**: `docs/training_guide.md`
- **API reference**: `docs/api_reference.md`
