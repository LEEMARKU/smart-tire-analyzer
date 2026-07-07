from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("model_selector")

TIRE_CONDITIONS = ["safe", "moderate", "replace"]
WEAR_TYPES = [
    "center_wear",
    "edge_wear",
    "patchy_wear",
    "uniform_wear",
    "one_side_wear",
    "cupping_wear",
]

from ai_model.models.model_factory import list_available_models, select_models, list_cnn_models, list_rnn_models, list_fusion_models


@dataclass
class DatasetReport:
    total_images: int
    train_count: int
    val_count: int
    test_count: int
    condition_distribution: dict[str, int]
    wear_distribution: dict[str, int]
    class_balance_score: float
    is_balanced: bool
    avg_samples_per_class: float
    has_front_images: bool
    has_sidewall_images: bool
    has_multi_view: bool

    @property
    def effective_train_size(self) -> int:
        return self.train_count


@dataclass
class RecommendedArchitecture:
    use_vit: bool
    cnn_model: str
    transformer_model: str | None
    rnn_model: str
    fusion_model: str
    cnn_feature_dim: int
    rnn_output_dim: int
    fusion_output_dim: int

    @property
    def total_feature_dim_before_fusion(self) -> int:
        vision_dim = self.cnn_feature_dim
        vit_dim = self.cnn_feature_dim if self.use_vit else 0
        return vision_dim + vit_dim + self.rnn_output_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_vit": self.use_vit,
            "cnn_model": self.cnn_model,
            "transformer_model": self.transformer_model,
            "rnn_model": self.rnn_model,
            "fusion_model": self.fusion_model,
            "cnn_feature_dim": self.cnn_feature_dim,
            "rnn_output_dim": self.rnn_output_dim,
            "fusion_output_dim": self.fusion_output_dim,
        }


@dataclass
class RecommendedHyperparams:
    batch_size: int
    stage1_epochs: int
    stage2_epochs: int
    learning_rate: float
    fine_tune_lr: float
    weight_decay: float
    grad_accum_steps: int
    dropout_rate: float
    patience: int
    augmentation_severity: str
    use_mixup: bool
    use_kfold: bool
    k_folds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "stage1_epochs": self.stage1_epochs,
            "stage2_epochs": self.stage2_epochs,
            "learning_rate": self.learning_rate,
            "fine_tune_lr": self.fine_tune_lr,
            "weight_decay": self.weight_decay,
            "grad_accum_steps": self.grad_accum_steps,
            "dropout_rate": self.dropout_rate,
            "patience": self.patience,
            "augmentation_severity": self.augmentation_severity,
            "use_mixup": self.use_mixup,
            "use_kfold": self.use_kfold,
            "k_folds": self.k_folds,
        }


@dataclass
class ModelRecommendation:
    dataset: DatasetReport
    architecture: RecommendedArchitecture
    hyperparams: RecommendedHyperparams
    dataset_category: str
    reasoning: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_category": self.dataset_category,
            "reasoning": self.reasoning,
            "dataset": {
                "total_images": self.dataset.total_images,
                "train_count": self.dataset.train_count,
                "val_count": self.dataset.val_count,
                "test_count": self.dataset.test_count,
                "condition_distribution": self.dataset.condition_distribution,
                "wear_distribution": self.dataset.wear_distribution,
                "class_balance_score": round(self.dataset.class_balance_score, 3),
                "is_balanced": self.dataset.is_balanced,
            },
            "architecture": self.architecture.to_dict(),
            "hyperparams": self.hyperparams.to_dict(),
        }


def _parse_condition_id(value: Any) -> str:
    try:
        idx = int(value)
        if 0 <= idx < len(TIRE_CONDITIONS):
            return TIRE_CONDITIONS[idx]
    except (ValueError, TypeError):
        pass
    label = str(value).strip().lower()
    if label in TIRE_CONDITIONS:
        return label
    return "unknown"


def _parse_wear_label(value: Any) -> str:
    label = str(value).strip().lower().replace(" ", "_")
    if label in WEAR_TYPES:
        return label
    for alias, canonical in [
        ("even", "uniform_wear"),
        ("even_wear", "uniform_wear"),
        ("patch_wear", "patchy_wear"),
        ("one_sided_wear", "one_side_wear"),
        ("cupping", "cupping_wear"),
        ("uneven_wear", "patchy_wear"),
        ("critical_wear", "patchy_wear"),
    ]:
        if label == alias:
            return canonical
    return "unknown"


def _images_in_dir(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _multi_view_count(project_root: Path) -> int:
    total = 0
    for sub in ("front_view", "side_view", "sidewall"):
        total += _images_in_dir(project_root / "dataset" / "multi_view" / "train" / sub)
    return total


def analyze_dataset(project_root: Path) -> DatasetReport:
    split_root = project_root / "dataset" / "splits"
    train_frame = _load_split_or_empty(split_root / "train")
    val_frame = _load_split_or_empty(split_root / "validation")
    test_frame = _load_split_or_empty(split_root / "test")

    train_count = len(train_frame)
    val_count = len(val_frame)
    test_count = len(test_frame)
    total = train_count + val_count + test_count

    condition_dist: dict[str, int] = {}
    wear_dist: dict[str, int] = {}
    for frame in (train_frame, val_frame, test_frame):
        if "condition_id" in frame.columns:
            for c in frame["condition_id"].apply(_parse_condition_id):
                condition_dist[c] = condition_dist.get(c, 0) + 1
        if "wear_pattern" in frame.columns:
            for w in frame["wear_pattern"].apply(_parse_wear_label):
                wear_dist[w] = wear_dist.get(w, 0) + 1

    class_counts = list(condition_dist.values())
    class_balance_score = 1.0
    if class_counts:
        mean_c = float(np.mean(class_counts))
        std_c = float(np.std(class_counts))
        class_balance_score = 1.0 / (1.0 + std_c / max(mean_c, 1))

    is_balanced = class_balance_score > 0.6
    avg_per_class = int(np.mean(class_counts)) if class_counts else 0

    has_front = _images_in_dir(project_root / "dataset" / "images" / "front_view") > 0
    has_sidewall = _images_in_dir(project_root / "dataset" / "images" / "sidewall") > 0
    has_multi_view = _multi_view_count(project_root) > 0

    return DatasetReport(
        total_images=total,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        condition_distribution=condition_dist,
        wear_distribution=wear_dist,
        class_balance_score=class_balance_score,
        is_balanced=is_balanced,
        avg_samples_per_class=avg_per_class,
        has_front_images=has_front,
        has_sidewall_images=has_sidewall,
        has_multi_view=has_multi_view,
    )


def _load_split_or_empty(split_dir: Path) -> pd.DataFrame:
    labels_path = split_dir / "labels.csv"
    if labels_path.exists():
        try:
            return pd.read_csv(labels_path)
        except Exception as exc:
            logger.warning("Could not read %s: %s", labels_path, exc)
    return pd.DataFrame()


def _categorize_dataset(train_count: int) -> str:
    if train_count < 200:
        return "tiny"
    if train_count < 500:
        return "very_small"
    if train_count < 2000:
        return "small"
    if train_count < 5000:
        return "small_plus"
    if train_count < 15000:
        return "medium"
    return "large"


def _recommend_architecture(report: DatasetReport) -> RecommendedArchitecture:
    config = select_models(report.train_count)
    use_vit = config["transformer"] is not None
    return RecommendedArchitecture(
        use_vit=use_vit,
        cnn_model=config["cnn"],
        transformer_model=config["transformer"],
        rnn_model=config["rnn"],
        fusion_model=config["fusion"],
        cnn_feature_dim=config["cnn_feature_dim"],
        rnn_output_dim=config["rnn_output_dim"],
        fusion_output_dim=config["fusion_output_dim"],
    )


def _recommend_hyperparams(report: DatasetReport) -> RecommendedHyperparams:
    category = _categorize_dataset(report.train_count)

    if category == "tiny":
        return RecommendedHyperparams(
            batch_size=2,
            stage1_epochs=50,
            stage2_epochs=0,
            learning_rate=3e-5,
            fine_tune_lr=1e-5,
            weight_decay=1e-2,
            grad_accum_steps=8,
            dropout_rate=0.5,
            patience=15,
            augmentation_severity="heavy",
            use_mixup=True,
            use_kfold=True,
            k_folds=5,
        )
    if category == "small":
        return RecommendedHyperparams(
            batch_size=4,
            stage1_epochs=30,
            stage2_epochs=10,
            learning_rate=5e-5,
            fine_tune_lr=1e-5,
            weight_decay=1e-3,
            grad_accum_steps=4,
            dropout_rate=0.4,
            patience=10,
            augmentation_severity="standard",
            use_mixup=True,
            use_kfold=False,
            k_folds=5,
        )
    if category == "medium":
        return RecommendedHyperparams(
            batch_size=8,
            stage1_epochs=20,
            stage2_epochs=15,
            learning_rate=1e-4,
            fine_tune_lr=1e-5,
            weight_decay=1e-3,
            grad_accum_steps=4,
            dropout_rate=0.3,
            patience=8,
            augmentation_severity="standard",
            use_mixup=True,
            use_kfold=False,
            k_folds=5,
        )
    return RecommendedHyperparams(
        batch_size=16,
        stage1_epochs=15,
        stage2_epochs=15,
        learning_rate=1e-4,
        fine_tune_lr=5e-6,
        weight_decay=1e-4,
        grad_accum_steps=2,
        dropout_rate=0.3,
        patience=6,
        augmentation_severity="light",
        use_mixup=False,
        use_kfold=False,
        k_folds=5,
    )


def _build_reasoning(report: DatasetReport, arch: RecommendedArchitecture, category: str) -> str:
    parts = [
        f"Dataset category: '{category}' ({report.train_count} training samples).",
    ]
    if arch.transformer_model is None:
        parts.append(f"Transformer disabled ({arch.cnn_model} handles vision alone).")
    else:
        parts.append(f"Dataset large enough for {arch.transformer_model} fine-tuning.")
    parts.append(
        f"Selected CNN: {arch.cnn_model}, RNN: {arch.rnn_model}, Fusion: {arch.fusion_model}."
    )
    if not report.is_balanced:
        imbalance_detail = ", ".join(
            f"{k}: {v}" for k, v in sorted(report.condition_distribution.items())
        )
        parts.append(
            f"Class imbalance detected ({imbalance_detail})."
        )
    if report.train_count < 500:
        parts.append("Very small dataset - heavy augmentation, k-fold CV, higher dropout.")
    if report.effective_train_size < 1500:
        parts.append(
            "Stage 1 (frozen encoder) dominates to leverage pretrained features."
        )
    return " ".join(parts)


def recommend(project_root: Path) -> ModelRecommendation:
    report = analyze_dataset(project_root)
    category = _categorize_dataset(report.train_count)
    arch = _recommend_architecture(report)
    hparams = _recommend_hyperparams(report)
    reasoning = _build_reasoning(report, arch, category)
    return ModelRecommendation(
        dataset=report,
        architecture=arch,
        hyperparams=hparams,
        dataset_category=category,
        reasoning=reasoning,
    )


def print_recommendation(recommendation: ModelRecommendation) -> None:
    print("=" * 65)
    print("  SMART TIRE ANALYZER - MODEL RECOMMENDATION ENGINE")
    print("=" * 65)
    print(f"\n  Dataset Category: [{recommendation.dataset_category.upper()}]")
    print(f"  {'=' * 50}")
    print(f"  Training samples:   {recommendation.dataset.train_count}")
    print(f"  Validation samples: {recommendation.dataset.val_count}")
    print(f"  Test samples:       {recommendation.dataset.test_count}")
    print(f"  Total images:       {recommendation.dataset.total_images}")
    if recommendation.dataset.condition_distribution:
        print(f"  Condition classes:  {recommendation.dataset.condition_distribution}")
    if recommendation.dataset.wear_distribution:
        print(f"  Wear pattern dist:  {recommendation.dataset.wear_distribution}")
    print(f"  Class balance:      {'BALANCED' if recommendation.dataset.is_balanced else 'IMBALANCED'}")
    print(f"  Multi-view avail:   {'Yes' if recommendation.dataset.has_multi_view else 'No'}")
    print()

    arch = recommendation.architecture
    print("  -- Recommended Architecture --")
    print(f"  Transformer:         {arch.transformer_model or 'None (not suitable)'}")
    print(f"  CNN backbone:        {arch.cnn_model} (dim={arch.cnn_feature_dim})")
    print(f"  RNN encoder:         {arch.rnn_model} (dim={arch.rnn_output_dim})")
    print(f"  Fusion network:      {arch.fusion_model} (dim={arch.fusion_output_dim})")
    print()

    hp = recommendation.hyperparams
    print("  -- Recommended Hyperparameters --")
    print(f"  Batch size:          {hp.batch_size}")
    print(f"  Stage 1 epochs:      {hp.stage1_epochs} (frozen encoder)")
    print(f"  Stage 2 epochs:      {hp.stage2_epochs} (fine-tune last blocks)")
    print(f"  Learning rate:       {hp.learning_rate}")
    print(f"  Fine-tune LR:        {hp.fine_tune_lr}")
    print(f"  Weight decay:        {hp.weight_decay}")
    print(f"  Grad accum steps:    {hp.grad_accum_steps}")
    print(f"  Dropout:             {hp.dropout_rate}")
    print(f"  Patience:            {hp.patience}")
    print(f"  Augmentation:        {hp.augmentation_severity}")
    print(f"  Mixup:               {'Yes' if hp.use_mixup else 'No'}")
    print(f"  K-fold CV:           {'Yes (' + str(hp.k_folds) + '-fold)' if hp.use_kfold else 'No'}")
    print()

    print("  -- Reasoning --")
    print(f"  {recommendation.reasoning}")
    print("=" * 65)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rec = recommend(root)
    print_recommendation(rec)
