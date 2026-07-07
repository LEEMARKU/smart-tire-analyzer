from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelFamily(Enum):
    CNN = "cnn"
    TRANSFORMER = "transformer"
    RNN = "rnn"
    FUSION = "fusion"

    @property
    def android_label(self) -> str:
        return {"cnn": "CNN", "transformer": "Transformer", "rnn": "RNN", "fusion": "Fusion/ANN"}[self.value]


@dataclass
class AndroidModelInfo:
    name: str
    family: ModelFamily
    params_m: float
    size_mb: float
    ram_mb: int
    latency: str
    min_ram: int
    min_android: int
    feature_dim: int
    is_vision: bool = True
    requires_gpu: bool = False
    tflite_name: str = ""

    def __post_init__(self):
        if not self.tflite_name:
            safe = self.name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            self.tflite_name = f"{safe}.tflite"

    @property
    def is_quantized(self) -> bool:
        return "quant" in self.tflite_name or "int8" in self.tflite_name

    @property
    def ram_category(self) -> str:
        if self.ram_mb < 200:
            return "low"
        if self.ram_mb < 400:
            return "medium"
        return "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "params_m": self.params_m,
            "size_mb": self.size_mb,
            "ram_mb": self.ram_mb,
            "latency": self.latency,
            "min_ram": self.min_ram,
            "min_android": self.min_android,
            "feature_dim": self.feature_dim,
            "is_vision": self.is_vision,
            "requires_gpu": self.requires_gpu,
            "tflite_name": self.tflite_name,
        }


CNN_ANDROID_REGISTRY: dict[str, dict[str, Any]] = {
    "MobileNetV1": {"params_m": 4.2, "size_mb": 16, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 1024},
    "MobileNetV2": {"params_m": 3.5, "size_mb": 14, "ram_mb": 180, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 1280},
    "MobileNetV3": {"params_m": 5.4, "size_mb": 22, "ram_mb": 250, "latency": "fast", "min_ram": 3072, "min_android": 24, "feature_dim": 1280},
    "EfficientNet-B0": {"params_m": 5.3, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 24, "feature_dim": 1280},
    "EfficientNet-Lite0": {"params_m": 4.7, "size_mb": 18, "ram_mb": 250, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 1280},
    "EfficientNetV2-B0": {"params_m": 7.1, "size_mb": 28, "ram_mb": 350, "latency": "medium", "min_ram": 4096, "min_android": 26, "feature_dim": 1280},
    "SqueezeNet": {"params_m": 1.2, "size_mb": 5, "ram_mb": 150, "latency": "very_fast", "min_ram": 1024, "min_android": 21, "feature_dim": 512},
    "ShuffleNetV2": {"params_m": 2.3, "size_mb": 9, "ram_mb": 160, "latency": "very_fast", "min_ram": 1024, "min_android": 21, "feature_dim": 768},
    "NASNet-Mobile": {"params_m": 5.7, "size_mb": 23, "ram_mb": 280, "latency": "medium", "min_ram": 3072, "min_android": 24, "feature_dim": 768},
    "GhostNet": {"params_m": 5.2, "size_mb": 20, "ram_mb": 220, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 960},
    "MnasNet": {"params_m": 4.4, "size_mb": 17, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 1280},
    "ConvNeXt-Tiny": {"params_m": 28.0, "size_mb": 110, "ram_mb": 600, "latency": "slow", "min_ram": 6144, "min_android": 28, "feature_dim": 768},
}

TRANSFORMER_ANDROID_REGISTRY: dict[str, dict[str, Any]] = {
    "MobileViT": {"params_m": 5.6, "size_mb": 22, "ram_mb": 350, "latency": "medium", "min_ram": 3072, "min_android": 26, "feature_dim": 384},
    "LeViT": {"params_m": 7.8, "size_mb": 31, "ram_mb": 400, "latency": "medium", "min_ram": 4096, "min_android": 26, "feature_dim": 384},
    "TinyViT": {"params_m": 5.0, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 26, "feature_dim": 256},
    "DeiT-Tiny": {"params_m": 5.7, "size_mb": 23, "ram_mb": 320, "latency": "medium", "min_ram": 3072, "min_android": 26, "feature_dim": 256},
    "EfficientFormer": {"params_m": 6.2, "size_mb": 25, "ram_mb": 350, "latency": "medium", "min_ram": 4096, "min_android": 26, "feature_dim": 512},
    "EdgeNeXt": {"params_m": 5.0, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 26, "feature_dim": 384},
    "SwiftFormer": {"params_m": 3.5, "size_mb": 14, "ram_mb": 250, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 384},
    "PiT": {"params_m": 4.9, "size_mb": 19, "ram_mb": 280, "latency": "medium", "min_ram": 3072, "min_android": 24, "feature_dim": 256},
}

RNN_ANDROID_REGISTRY: dict[str, dict[str, Any]] = {
    "LSTM": {"params_m": 0.5, "size_mb": 2, "ram_mb": 80, "latency": "very_fast", "min_ram": 512, "min_android": 21, "feature_dim": 256},
    "BiLSTM": {"params_m": 1.0, "size_mb": 4, "ram_mb": 120, "latency": "very_fast", "min_ram": 1024, "min_android": 21, "feature_dim": 256},
    "GRU": {"params_m": 0.4, "size_mb": 2, "ram_mb": 80, "latency": "very_fast", "min_ram": 512, "min_android": 21, "feature_dim": 256},
    "BiGRU": {"params_m": 0.8, "size_mb": 3, "ram_mb": 100, "latency": "very_fast", "min_ram": 1024, "min_android": 21, "feature_dim": 256},
    "Simple RNN": {"params_m": 0.2, "size_mb": 1, "ram_mb": 50, "latency": "very_fast", "min_ram": 512, "min_android": 21, "feature_dim": 128},
    "TCN": {"params_m": 0.8, "size_mb": 3, "ram_mb": 100, "latency": "very_fast", "min_ram": 1024, "min_android": 21, "feature_dim": 256},
}

FUSION_ANDROID_REGISTRY: dict[str, dict[str, Any]] = {
    "MLP": {"params_m": 2.1, "size_mb": 8, "ram_mb": 150, "latency": "fast", "min_ram": 1024, "min_android": 21, "feature_dim": 512},
    "Deep Dense Fusion": {"params_m": 3.5, "size_mb": 14, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
    "Early Fusion": {"params_m": 0.5, "size_mb": 2, "ram_mb": 80, "latency": "very_fast", "min_ram": 512, "min_android": 21, "feature_dim": 512},
    "Late Fusion": {"params_m": 2.0, "size_mb": 8, "ram_mb": 160, "latency": "fast", "min_ram": 1024, "min_android": 24, "feature_dim": 512},
    "Hybrid Fusion": {"params_m": 2.8, "size_mb": 11, "ram_mb": 180, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
    "Cross-Modal Attention": {"params_m": 4.2, "size_mb": 17, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 26, "feature_dim": 512},
    "Attention-based Fusion": {"params_m": 2.5, "size_mb": 10, "ram_mb": 180, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
    "Gated Fusion": {"params_m": 2.2, "size_mb": 9, "ram_mb": 170, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
    "Residual Fusion": {"params_m": 2.0, "size_mb": 8, "ram_mb": 150, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
    "Adaptive Fusion": {"params_m": 3.0, "size_mb": 12, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24, "feature_dim": 512},
}

_DATASET_TIERS: list[tuple[str, int, int]] = [
    ("tiny", 0, 200),
    ("very_small", 200, 500),
    ("small", 500, 2000),
    ("medium", 2000, 5000),
    ("large", 5000, 15000),
    ("very_large", 15000, float("inf")),
]


def _get_dataset_tier(train_count: int) -> str:
    for name, lo, hi in _DATASET_TIERS:
        if lo <= train_count < hi:
            return name
    return "very_large"


def _get_device_tier(total_ram_mb: int, android_sdk: int, has_gpu: bool) -> str:
    if total_ram_mb >= 6144 and android_sdk >= 28 and has_gpu:
        return "high"
    if total_ram_mb >= 3072 and android_sdk >= 26:
        return "medium"
    if total_ram_mb >= 2048 and android_sdk >= 24:
        return "low_medium"
    return "low"


_CNN_BY_TIER: dict[str, list[str]] = {
    "low": ["SqueezeNet", "ShuffleNetV2"],
    "low_medium": ["MobileNetV1", "MobileNetV2", "EfficientNet-Lite0", "MnasNet"],
    "medium": ["MobileNetV3", "GhostNet", "NASNet-Mobile", "EfficientNet-B0"],
    "high": ["EfficientNetV2-B0", "ConvNeXt-Tiny"],
}

_TRANSFORMER_BY_TIER: dict[str, list[str]] = {
    "low": [],
    "low_medium": ["SwiftFormer"],
    "medium": ["MobileViT", "TinyViT", "DeiT-Tiny", "EdgeNeXt", "PiT"],
    "high": ["EfficientFormer", "LeViT"],
}

_RNN_BY_TIER: dict[str, list[str]] = {
    "low": ["Simple RNN", "LSTM", "GRU"],
    "low_medium": ["BiLSTM", "BiGRU", "TCN"],
    "medium": ["BiLSTM", "BiGRU", "TCN"],
    "high": ["BiLSTM", "BiGRU", "TCN"],
}

_FUSION_BY_TIER: dict[str, list[str]] = {
    "low": ["Early Fusion", "MLP"],
    "low_medium": ["Late Fusion", "Residual Fusion", "Gated Fusion"],
    "medium": ["Hybrid Fusion", "Attention-based Fusion", "Deep Dense Fusion", "Adaptive Fusion"],
    "high": ["Cross-Modal Attention", "Adaptive Fusion"],
}


_ALL_MODELS: dict[str, AndroidModelInfo] = {}


def _populate():
    for family, registry, is_vision in [
        (ModelFamily.CNN, CNN_ANDROID_REGISTRY, True),
        (ModelFamily.TRANSFORMER, TRANSFORMER_ANDROID_REGISTRY, True),
        (ModelFamily.RNN, RNN_ANDROID_REGISTRY, False),
        (ModelFamily.FUSION, FUSION_ANDROID_REGISTRY, False),
    ]:
        for name, spec in registry.items():
            _ALL_MODELS[name] = AndroidModelInfo(
                name=name,
                family=family,
                is_vision=is_vision,
                **spec,
            )


_populate()


@dataclass
class AndroidModelRecommendation:
    cnn_model: AndroidModelInfo
    transformer_model: AndroidModelInfo | None
    rnn_model: AndroidModelInfo
    fusion_model: AndroidModelInfo
    device_tier: str
    dataset_tier: str
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cnn": self.cnn_model.to_dict(),
            "transformer": self.transformer_model.to_dict() if self.transformer_model else None,
            "rnn": self.rnn_model.to_dict(),
            "fusion": self.fusion_model.to_dict(),
            "device_tier": self.device_tier,
            "dataset_tier": self.dataset_tier,
            "reasoning": self.reasoning,
        }


class AndroidModelRegistry:
    @staticmethod
    def list_all() -> list[AndroidModelInfo]:
        return list(_ALL_MODELS.values())

    @staticmethod
    def list_by_family(family: ModelFamily) -> list[AndroidModelInfo]:
        return [m for m in _ALL_MODELS.values() if m.family == family]

    @staticmethod
    def get(name: str) -> AndroidModelInfo | None:
        return _ALL_MODELS.get(name)

    @staticmethod
    def recommend(
        total_ram_mb: int,
        android_sdk: int,
        dataset_size: int,
        has_gpu: bool = False,
        has_nnapi: bool = False,
    ) -> AndroidModelRecommendation:
        device_tier = _get_device_tier(total_ram_mb, android_sdk, has_gpu or has_nnapi)
        dataset_tier = _get_dataset_tier(dataset_size)

        cnn_candidates = _CNN_BY_TIER.get(device_tier, _CNN_BY_TIER["low_medium"])
        transformer_candidates = _TRANSFORMER_BY_TIER.get(device_tier, [])
        rnn_candidates = _RNN_BY_TIER.get(device_tier, _RNN_BY_TIER["low"])
        fusion_candidates = _FUSION_BY_TIER.get(device_tier, _FUSION_BY_TIER["low"])

        cnn_model = _pick_best(cnn_candidates, dataset_tier, ModelFamily.CNN)
        transformer_model = _pick_best(transformer_candidates, dataset_tier, ModelFamily.TRANSFORMER) if transformer_candidates else None
        rnn_model = _pick_best(rnn_candidates, dataset_tier, ModelFamily.RNN)
        fusion_model = _pick_best(fusion_candidates, dataset_tier, ModelFamily.FUSION)

        parts = [
            f"Device tier: '{device_tier}' ({total_ram_mb}MB RAM, SDK {android_sdk}).",
            f"Dataset tier: '{dataset_tier}' ({dataset_size} samples).",
        ]
        if transformer_model is None:
            parts.append("Transformer disabled - device too constrained.")
        else:
            parts.append("Device capable of transformer model.")
        parts.append(f"CNN: {cnn_model.name}, Transformer: {transformer_model.name if transformer_model else 'None'}, "
                     f"RNN: {rnn_model.name}, Fusion: {fusion_model.name}.")

        return AndroidModelRecommendation(
            cnn_model=cnn_model,
            transformer_model=transformer_model,
            rnn_model=rnn_model,
            fusion_model=fusion_model,
            device_tier=device_tier,
            dataset_tier=dataset_tier,
            reasoning=" ".join(parts),
        )


def _pick_best(candidates: list[str], dataset_tier: str, family: ModelFamily) -> AndroidModelInfo:
    available = [m for m in (_ALL_MODELS.get(n) for n in candidates) if m is not None]
    if not available:
        all_of_family = AndroidModelRegistry.list_by_family(family)
        available = all_of_family

    tier_scores = {"tiny": 0, "very_small": 1, "small": 2, "medium": 3, "large": 4, "very_large": 5}
    score = tier_scores.get(dataset_tier, 3)

    if score <= 1:
        available.sort(key=lambda m: (m.params_m, m.ram_mb))
    elif score <= 3:
        available.sort(key=lambda m: m.params_m)
    else:
        available.sort(key=lambda m: -m.feature_dim)

    idx = min(score, len(available) - 1) if available else 0
    return available[max(0, idx)]
