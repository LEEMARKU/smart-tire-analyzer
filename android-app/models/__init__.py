"""
Android-ready model definitions for Smart Tire Analyzer.

These models extend the ai_model registry with additional architectures
optimized for mobile/edge deployment. Run from the project root:

    from ai_model.models.cnn_backbones import CNNBackboneFactory
    import android_app.models.android_cnn_models  # registers new models

    cnn = CNNBackboneFactory.create("MobileNetV1")
"""

import sys
from pathlib import Path

_ANDROID_MODELS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ANDROID_MODELS_DIR.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
