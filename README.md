# Smart Tire Analyzer

<div align="center">
  <h3>AI-Powered Cross-Platform Tire Intelligence System</h3>
  <p>
    59 Model Architectures · Auto Model Selection · CNN + RNN + ANN · Gemini AI Reasoning
  </p>
  <p>
    <strong>Live Chat</strong> (Llama 3.3 tire-only AI) · <strong>Voice AI Support</strong> (OmniDimension)
  </p>
</div>

---

## Overview

**Smart Tire Analyzer** analyzes tire condition from photographs using a hybrid deep learning model with **automatic architecture selection**. It supports 59 model variants (22 CNN, 12 Transformer, 9 RNN, 16 Fusion/ANN) and chooses the optimal combination based on dataset size. It predicts tread depth, health score, remaining life, condition (safe/moderate/replace), and wear pattern.<br><br>
The project also includes a **Next.js frontend** with a Live Chat feature (Llama 3.3, tire-only questions) and a **Voice AI Support** agent (OmniDimension, Llama 3.3 70B) for tire and technical support calls.

### Current Test Performance (767-image dataset)

| Metric | Accuracy |
|--------|:--------:|
| **Condition** (safe/moderate/replace) | **90.7%** |
| **Wear Pattern** (6 classes) | **61.7%** |
| **Tread Depth MAE** | **1.55 mm** |
| **Health Score MAE** | 0.43 |
| **Remaining Life MAE** | ~3,964 km |

### Key Features

| Feature | Detail |
|---------|--------|
| **Auto Model Selection** | Picks optimal CNN + RNN + ANN by dataset size (6 tiers) |
| **59 Model Architectures** | 22 CNN · 12 Transformer · 9 RNN · 16 Fusion/ANN |
| **Tread Depth Prediction** | 4-point measurement (T1–T4) |
| **Condition Classification** | 3 classes: safe, moderate, replace |
| **Wear Pattern Detection** | 6 classes: center, edge, patchy, uniform, one-side, cupping |
| **24-Step Preprocessing** | Auto-rotate, shadow removal, GrabCut, CLAHE, edge detect, etc. |
| **Class-Weighted Training** | Inverse-frequency weighting for imbalanced classes |
| **Gemini AI Reasoning** | Context-aware driving advice and replacement urgency |
| **Continuous Learning** | User corrections → auto-retrain after 10 samples |
| **Live Chat** | Llama 3.3 tire-only AI assistant on `/live-chat` |
| **Voice AI Support** | OmniDimension voice agent (Llama 3.3 70B) on `/technical-support` |

---

## Architecture

```
📷 Tire Image
       ↓
[24-Step Preprocessing Pipeline]
  ├── Auto rotation correction (deskew via Hough lines)
  ├── Shadow removal (LAB illumination estimation)
  ├── Background removal (Hough circle + GrabCut refinement)
  ├── Blur detection + deblurring
  ├── CLAHE contrast enhancement
  ├── Edge detection (4th channel)
  ├── Resize (224x224) + ImageNet normalization
  └── Augmentation (training only)
       ↓
[Auto-Selected Architecture — example for 767 samples]
  ├── CNN: EfficientNetV2-B0 → 512-dim local tread features
  ├── RNN: BiLSTM → 256-dim sequential tread features
  └── Fusion: Deep Dense ANN → 512-dim fused features
       ↓
[Prediction Heads]
  ├── Tread Depth (×4 regression)
  ├── Health Score (regression)
  ├── Remaining Life (regression)
  ├── Wear Pattern (6-class classification)
  └── Condition (3-class: safe/moderate/replace)
       ↓
[Gemini AI Reasoning + Maps + Weather]
       ↓
📊 Final Report
```

Model auto-selection tiers:

| Dataset Size | Tier | CNN | RNN | Fusion | Transformer |
|---|---|---|---|---|---|
| 0–200 | tiny | ResNet18 | GRU | Standard FC | None |
| 200–500 | very_small | MobileNetV2 | BiGRU | MLP | None |
| 500–2000 | small | EfficientNetV2-B0 | BiLSTM | Deep Dense Fusion | None |
| 2000–5000 | small_plus | EfficientNetV2-B0 | Stacked LSTM | Self-Attention Fusion | None |
| 5000–15000 | medium | ConvNeXt | Encoder-Decoder LSTM | Cross-Modal Attention | ViT (optional) |
| 15000+ | large | ConvNeXt | TCN | Multimodal Transformer Fusion | ViT |

---

## Quick Start

### 1. Setup Environment

```bash
git clone <repo-url>
cd smart-tire-analyzer

# Create virtual environment + install deps
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r backend/requirements.txt
pip install -r ai_model/hybrid_torch/requirements.txt
```

### 2. Place Dataset

Add tire images with labels to `dataset/splits/`:

```
dataset/splits/
├── train/labels.csv      # Training set
├── validation/labels.csv # Validation set
└── test/labels.csv       # Test set
```

Each CSV must include `image_path`, `condition_id` (0=safe,1=moderate,2=replace), `wear_pattern`, and tread columns (`tread_1`–`tread_4`).

### 3. Smart Training (Auto Model Selection)

```bash
# Full training with auto-config
python scripts/train_smart.py

# Analysis only (no training)
python scripts/train_smart.py --analyze
```

The system will:
- Count your samples and classify into a tier
- Select optimal CNN + RNN + Fusion (and Transformer if enough data)
- Train in two stages: frozen encoder (30 epochs) → fine-tune (25 epochs)
- Apply class-weighted loss to handle imbalance
- Save best checkpoint to `ai_model/saved_models/hybrid_torch/`

### 4. Start Backend

```bash
python scripts/start_server.py
```

**API:** `http://localhost:8000`  
**Swagger:** `http://localhost:8000/docs`

### 5. Start Frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

**Frontend:** `http://localhost:3000`  
**Live Chat:** `http://localhost:3000/live-chat`  
**Technical Support:** `http://localhost:3000/technical-support` (with Voice AI)

### 6. Test Inference

```bash
# Analyze a tire image
python scripts/infer.py --image path/to/tire.jpg

# With GPS context
python scripts/infer.py --image tire.jpg --lat 28.61 --lon 77.21
```

Or double-click `run_services.bat` and choose **Local Dev Mode**.

---

## Training Details

### Loss Function

Multi-task loss with class-weighted cross-entropy for imbalance:

| Component | Weight | Notes |
|-----------|:------:|-------|
| Tread Depth L1 | 3.5 | Smooth L1 + extra penalty for >1mm error |
| Health MSE | 0.7 | |
| Remaining Life MSE | 0.7 | |
| Wear Pattern CE | 1.0 | Inverse-frequency class weighted |
| Condition CE | 1.0 | Inverse-frequency class weighted |

Class weights are computed from the training set distribution (e.g., "replace" condition gets ~3.5× weight vs "safe").

### Training Stages

1. **Stage 1** (30 epochs): Frozen pretrained encoder, trains heads + fusion + RNN
2. **Stage 2** (25 epochs): Fine-tune last 3 CNN blocks + all heads

### Outputs

```
ai_model/saved_models/hybrid_torch/
├── model_best.pt            # Best checkpoint (by tread MAE)
├── model_last.pt            # Last checkpoint
├── metadata.json            # Architecture + hyperparams + calibration
├── metrics.json             # Validation/test metrics
├── history.json             # Per-epoch training history
└── tread_calibration.json   # Isotonic regression calibrator
```

---

## Docker Deployment

```bash
docker compose -f deployment/docker/docker-compose.yml up --build -d
curl http://localhost:8000/health
```

---

## API Reference

### `POST /analyze`

Upload a tire image → analysis report.

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | ✅ | JPEG/PNG, max 10MB |
| `latitude` | float | ❌ | GPS for road context |
| `longitude` | float | ❌ | GPS for road context |
| `tire_brand` | string | ❌ | e.g. "Michelin" |
| `mileage_km` | float | ❌ | Current mileage |

### `POST /feedback`

Submit user correction for continuous learning.

```json
{
  "session_id": "uuid",
  "feedback_type": "wrong",
  "corrected_tread_depth_mm": 4.5,
  "corrected_wear_pattern": "edge_wear"
}
```

### `GET /history`

Paginated analysis history.

### `GET /health`

Service health check.

---

## Project Structure

```
smart-tire-analyzer/
├── ai_model/
│   ├── models/            # 59 model implementations (CNN, RNN, Fusion)
│   ├── hybrid_torch/      # Training pipeline, dataset, evaluation
│   │   ├── trainer.py     # Multi-task training with class weighting
│   │   ├── model.py       # Model architecture + checkpoint loading
│   │   ├── dataset.py     # HybridTireDataset with tread sequences
│   │   └── constants.py   # Labels, aliases, hyperparams
│   ├── model_selector.py  # Auto-architecture recommender
│   └── saved_models/      # Trained model checkpoints
├── backend/
│   └── app/
│       ├── routes/        # /analyze, /feedback, /history, /health, /support
│       ├── services/      # Inference, Gemini, Maps, Weather, Omnidim
│       └── models/        # Pydantic schemas
├── frontend/
│   └── app/
│       ├── api/live-chat/ # Llama 3.3 tire-only AI assistant API
│       ├── live-chat/     # Live Chat page
│       ├── contact/       # Contact page
│       ├── technical-support/ # Technical Support + Voice AI
│       └── ...            # Other pages (home, documentation, etc.)
├── dataset/
│   ├── splits/            # train/val/test CSVs with image paths
│   ├── preprocessing/     # Pipeline: rotation, shadow, GrabCut, etc.
│   └── raw/               # Raw tread images
├── scripts/
│   ├── train_smart.py     # Smart training entry point
│   ├── start_server.py    # Backend API server
│   ├── infer.py           # Local inference
│   └── setup_env.py       # Environment setup
├── deployment/
│   └── docker/            # Dockerfile + compose
├── debug_root.bat         # System diagnostics
├── run_services.bat       # Interactive launcher (Windows)
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
