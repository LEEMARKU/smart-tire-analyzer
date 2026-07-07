"""
Smart Tire Analyzer - Consolidated Training Pipeline
=====================================================
1. Auto-configures model architecture based on dataset size + device specs
2. Trains the recommended CNN + Transformer + RNN + Fusion pipeline
3. Exports to TFLite (FP16 + INT8) for Android deployment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Register all Android-optimized models (MobileNetV1, ShuffleNetV2, MobileViT, etc.)
try:
    import android_app.models.android_cnn_models  # registers new CNN models
    import android_app.models.android_transformer_models  # registers new transformer models
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_new_models")


def delete_old_artifacts():
    """Clean all old trained models before fresh training."""
    paths_to_delete = [
        _PROJECT_ROOT / "ai_model" / "saved_models" / "hybrid_torch" / "model_best.pt",
        _PROJECT_ROOT / "ai_model" / "saved_models" / "hybrid_torch" / "model_last.pt",
        _PROJECT_ROOT / "ai_model" / "model_archive",
    ]
    for path in paths_to_delete:
        if path.is_file():
            path.unlink()
            logger.info("Deleted old: %s", path)
        elif path.is_dir():
            import shutil
            shutil.rmtree(path)
            logger.info("Deleted old archive: %s")

    assets = _PROJECT_ROOT / "android-app" / "app" / "src" / "main" / "assets"
    for tflite in assets.glob("*.tflite"):
        tflite.unlink()
        logger.info("Deleted old TFLite: %s", tflite)


def analyze_dataset():
    """Run model selector to determine the best architecture."""
    from ai_model.model_selector import recommend, print_recommendation
    recommendation = recommend(_PROJECT_ROOT)
    print_recommendation(recommendation)
    return recommendation


def train_hybrid_model(recommendation, epochs: int | None = None):
    """Train the auto-configured hybrid model."""
    from ai_model.hybrid_torch.trainer import FreshHybridConfig, train_fresh_hybrid

    arch = recommendation.architecture
    hp = recommendation.hyperparams

    config = FreshHybridConfig(
        project_root=_PROJECT_ROOT,
        auto_configure=True,
        archive_old=False,
        stage1_epochs=epochs or hp.stage1_epochs,
        stage2_epochs=max(epochs or hp.stage2_epochs, 25),
        batch_size=hp.batch_size,
        grad_accum_steps=hp.grad_accum_steps,
        learning_rate=hp.learning_rate,
        fine_tune_learning_rate=hp.fine_tune_lr * 3.0,
        weight_decay=hp.weight_decay,
        patience=hp.patience,
        use_vit=arch.use_vit,
        cnn_feature_dim=arch.cnn_feature_dim,
        rnn_output_dim=arch.rnn_output_dim,
        fusion_output_dim=arch.fusion_output_dim,
        pretrained_required=True,
    )

    logger.info("=" * 65)
    logger.info("Starting fresh hybrid model training")
    logger.info("  Architecture: CNN=%s, Transformer=%s, RNN=%s, Fusion=%s",
                 arch.cnn_model, arch.transformer_model or "None",
                 arch.rnn_model, arch.fusion_model)
    logger.info("  Hyperparams: batch=%d, epochs=%d+%d, lr=%s, decay=%s",
                 hp.batch_size, hp.stage1_epochs, hp.stage2_epochs,
                 hp.learning_rate, hp.weight_decay)
    logger.info("=" * 65)

    result = train_fresh_hybrid(config)
    logger.info("Training complete. Test metrics: %s",
                json.dumps(result["metrics"]["test"], indent=2))
    return result


def export_to_tflite():
    """Export trained PyTorch model to TFLite (FP16 + INT8) for Android."""
    from ai_model.hybrid_torch.model import HybridTireModel, build_model_from_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best checkpoint
    checkpoint_path = _PROJECT_ROOT / "ai_model" / "saved_models" / "hybrid_torch" / "model_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = _PROJECT_ROOT / "ai_model" / "saved_models" / "hybrid_torch" / "model_last.pt"
    if not checkpoint_path.exists():
        logger.warning("No trained checkpoint found; exporting untrained model")
        model = HybridTireModel(pretrained=False)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = build_model_from_checkpoint(checkpoint, device)

    model.eval()
    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_tread = torch.randn(1, 4, 7, device=device)

    # Export to ONNX
    import torch.onnx
    assets_dir = _PROJECT_ROOT / "android-app" / "app" / "src" / "main" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = assets_dir / "model.onnx"
    torch.onnx.export(
        model,
        {"image": dummy_image, "tread_sequence": dummy_tread},
        onnx_path,
        input_names=["image", "tread_sequence"],
        output_names=["tread_depths", "health_score", "remaining_life", "wear_pattern", "condition"],
        opset_version=12,
        dynamic_axes={
            "image": {0: "batch_size"},
            "tread_sequence": {0: "batch_size"},
        },
    )
    logger.info("Exported ONNX: %s", onnx_path)

    # Convert to TFLite via onnx-tf
    try:
        import onnx
        import tensorflow as tf

        onnx_model = onnx.load(str(onnx_path))
        import onnx_tf
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_path = assets_dir / "saved_model"
        if tf_path.exists():
            import shutil
            shutil.rmtree(tf_path)
        tf_rep.export_graph(str(tf_path))

        # FP16
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        fp16_tflite = assets_dir / "model_fp16.tflite"
        fp16_tflite.write_bytes(converter.convert())
        logger.info("Exported FP16 TFLite: %s (%.1f MB)", fp16_tflite, fp16_tflite.stat().st_size / 1e6)

        # INT8
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: [
            {"image": np.random.randn(1, 224, 224, 3).astype(np.float32),
             "tread_sequence": np.random.randn(1, 4, 7).astype(np.float32)}
        ]
        converter.target_spec.supported_types = [tf.int8]
        int8_tflite = assets_dir / "model_int8.tflite"
        int8_tflite.write_bytes(converter.convert())
        logger.info("Exported INT8 TFLite: %s (%.1f MB)", int8_tflite, int8_tflite.stat().st_size / 1e6)

        # FP32
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        fp32_tflite = assets_dir / "model_latest.tflite"
        fp32_tflite.write_bytes(converter.convert())
        logger.info("Exported FP32 TFLite: %s (%.1f MB)", fp32_tflite, fp32_tflite.stat().st_size / 1e6)
    except ImportError as e:
        logger.warning("TFLite export skipped (%s). Install onnx, onnx_tf, tensorflow", e)


def write_android_registry():
    """Generate Android model registry JSON for the trained models."""
    assets_dir = _PROJECT_ROOT / "android-app" / "app" / "src" / "main" / "assets"
    registry = {
        "models": [
            {"name": "model.onnx", "type": "ONNX", "description": "ONNX Runtime (primary)"},
            {"name": "model_latest.tflite", "type": "FP32", "description": "Full precision"},
            {"name": "model_fp16.tflite", "type": "FP16", "description": "Half precision (balanced)"},
            {"name": "model_int8.tflite", "type": "INT8", "description": "Quantized (fastest)"},
        ],
        "recommended": "model.onnx",
    }
    (assets_dir / "model_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    logger.info("Wrote Android model registry: %s", assets_dir / "model_registry.json")


def main():
    parser = argparse.ArgumentParser(description="Train new recommended models for Smart Tire Analyzer")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (stage1). Stage2 = max(V, 25)")
    parser.add_argument("--skip-tflite", action="store_true",
                        help="Skip TFLite export")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only analyze dataset, don't train")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("  SMART TIRE ANALYZER - NEW MODEL TRAINING PIPELINE")
    logger.info("=" * 65)

    # Step 1: Clean old artifacts
    logger.info("[1/5] Deleting old trained models...")
    delete_old_artifacts()

    # Step 2: Analyze dataset & get recommendation
    logger.info("[2/5] Analyzing dataset for model selection...")
    recommendation = analyze_dataset()

    if args.dry_run:
        logger.info("Dry run complete. No training performed.")
        return

    # Step 3: Train new model
    logger.info("[3/5] Training new recommended model architecture...")
    start = time.time()
    result = train_hybrid_model(recommendation, epochs=args.epochs)
    elapsed = time.time() - start
    logger.info("Training finished in %.1f minutes", elapsed / 60)

    # Step 4: Export to TFLite
    if not args.skip_tflite:
        logger.info("[4/5] Exporting to TFLite for Android...")
        export_to_tflite()

    # Step 5: Write registry
    logger.info("[5/5] Writing Android model registry...")
    write_android_registry()

    logger.info("=" * 65)
    logger.info("  ALL DONE - New models trained and ready for Android")
    logger.info("  Best val_loss: %.4f", result["metrics"]["validation"]["loss"])
    logger.info("  Test tread MAE: %.4f mm", result["metrics"]["test"]["tread_mae_mm"])
    logger.info("=" * 65)


if __name__ == "__main__":
    import numpy as np
    import torch
    main()
