from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ai_model.models.cnn_backbones import CNNBackboneFactory
from ai_model.models.transformer_backbones import TransformerBackboneFactory
from ai_model.models.rnn_encoders import RNNEncoderFactory
from ai_model.models.fusion_networks import FusionNetworkFactory

# Register Android-optimized mobile models from android-app/models/
import importlib.util
import sys
_android_models_dir = Path(__file__).resolve().parent.parent.parent / "android-app" / "models"
if _android_models_dir.is_dir() and str(_android_models_dir) not in sys.path:
    sys.path.insert(0, str(_android_models_dir))
    for _mod_name in ("android_cnn_models", "android_transformer_models"):
        try:
            importlib.import_module(_mod_name)
        except Exception:
            pass

logger = logging.getLogger("model_factory")

TIRE_CONDITION_CLASSES = 3
WEAR_PATTERN_CLASSES = 6


def list_cnn_models() -> list[str]:
    return CNNBackboneFactory.list_models()


def list_transformer_models() -> list[str]:
    return TransformerBackboneFactory.list_models()


def list_rnn_models() -> list[str]:
    return RNNEncoderFactory.list_models()


def list_fusion_models() -> list[str]:
    return FusionNetworkFactory.list_models()


def list_available_models() -> dict[str, list[str]]:
    return {
        "cnn": list_cnn_models(),
        "transformer": list_transformer_models(),
        "rnn": list_rnn_models(),
        "fusion": list_fusion_models(),
    }


DATASET_CATEGORY_RECOMMENDATIONS: dict[str, dict[str, Any]] = {
    "tiny": {
        "cnn": "SqueezeNet",
        "transformer": None,
        "rnn": "Simple RNN",
        "fusion": "Early Fusion",
        "cnn_feature_dim": 128,
        "rnn_output_dim": 128,
        "fusion_output_dim": 256,
    },
    "very_small": {
        "cnn": "MobileNetV1",
        "transformer": None,
        "rnn": "LSTM",
        "fusion": "Late Fusion",
        "cnn_feature_dim": 256,
        "rnn_output_dim": 128,
        "fusion_output_dim": 256,
    },
    "small": {
        "cnn": "ShuffleNetV2",
        "transformer": None,
        "rnn": "BiLSTM",
        "fusion": "Gated Fusion",
        "cnn_feature_dim": 512,
        "rnn_output_dim": 256,
        "fusion_output_dim": 512,
    },
    "small_plus": {
        "cnn": "EfficientNet-Lite0",
        "transformer": "SwiftFormer",
        "rnn": "BiLSTM",
        "fusion": "Hybrid Fusion",
        "cnn_feature_dim": 512,
        "rnn_output_dim": 256,
        "fusion_output_dim": 512,
    },
    "medium": {
        "cnn": "MobileNetV3",
        "transformer": "MobileViT",
        "rnn": "BiGRU",
        "fusion": "Deep Dense Fusion",
        "cnn_feature_dim": 1024,
        "rnn_output_dim": 256,
        "fusion_output_dim": 512,
    },
    "large": {
        "cnn": "ConvNeXt-Tiny",
        "transformer": "EfficientFormer",
        "rnn": "Stacked LSTM",
        "fusion": "Cross-Modal Attention",
        "cnn_feature_dim": 768,
        "rnn_output_dim": 512,
        "fusion_output_dim": 1024,
    },
}

TIERS = [
    ("tiny", 0, 200),
    ("very_small", 200, 500),
    ("small", 500, 2000),
    ("small_plus", 2000, 5000),
    ("medium", 5000, 15000),
    ("large", 15000, float("inf")),
]


def select_models(dataset_size: int) -> dict[str, Any]:
    for category, lo, hi in TIERS:
        if lo <= dataset_size < hi:
            config = dict(DATASET_CATEGORY_RECOMMENDATIONS[category])
            config["dataset_category"] = category
            return config
    config = dict(DATASET_CATEGORY_RECOMMENDATIONS["large"])
    config["dataset_category"] = "large"
    return config


def build_model_component(
    component_type: str,
    model_name: str,
    pretrained: bool = True,
    feature_dim: int | None = None,
    input_dim: int = 7,
    output_dim: int | None = None,
    cnn_dim: int = 512,
    vit_dim: int = 0,
    rnn_dim: int = 256,
) -> nn.Module:
    component_type = component_type.strip().lower()
    if component_type == "cnn":
        return CNNBackboneFactory.create(model_name, pretrained=pretrained, feature_dim=feature_dim)
    if component_type == "transformer":
        if model_name is None:
            return None
        return TransformerBackboneFactory.create(model_name, pretrained=pretrained, feature_dim=feature_dim)
    if component_type == "rnn":
        return RNNEncoderFactory.create(model_name, input_dim=input_dim, output_dim=output_dim)
    if component_type == "fusion":
        return FusionNetworkFactory.create(model_name, cnn_dim=cnn_dim, vit_dim=vit_dim, rnn_dim=rnn_dim, output_dim=output_dim or 512)
    raise ValueError(f"Unknown component type: {component_type}. Use: cnn, transformer, rnn, fusion")


class SmartTireModel(nn.Module):
    def __init__(self, cnn_name: str = "EfficientNetV2-B0", transformer_name: str | None = None, rnn_name: str = "BiLSTM", fusion_name: str = "Deep Dense Fusion", pretrained: bool = True, cnn_feature_dim: int = 512, rnn_output_dim: int = 256, fusion_output_dim: int = 512, tread_input_dim: int = 7):
        super().__init__()
        self.cnn = build_model_component("cnn", cnn_name, pretrained=pretrained, feature_dim=cnn_feature_dim)
        self.cnn_feature_dim = cnn_feature_dim
        self.transformer_name = transformer_name
        if transformer_name is not None:
            transformer_info = TransformerBackboneFactory.get_info(transformer_name)
            transformer_dim = transformer_info["feature_dim"]
            self.transformer = build_model_component("transformer", transformer_name, pretrained=pretrained, feature_dim=cnn_feature_dim)
        else:
            self.transformer = None
            transformer_dim = 0
        self.rnn = build_model_component("rnn", rnn_name, input_dim=tread_input_dim, output_dim=rnn_output_dim)
        self.rnn_output_dim = rnn_output_dim
        self.fusion = build_model_component("fusion", fusion_name, cnn_dim=cnn_feature_dim, vit_dim=cnn_feature_dim if transformer_name else 0, rnn_dim=rnn_output_dim, output_dim=fusion_output_dim)
        self.fusion_output_dim = fusion_output_dim
        vision_fused_dim = cnn_feature_dim + (cnn_feature_dim if transformer_name else 0)
        self.tread_head = nn.Sequential(
            nn.Linear(vision_fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        self.health_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.life_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.wear_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, WEAR_PATTERN_CLASSES),
        )
        self.condition_head = nn.Sequential(
            nn.Linear(fusion_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, TIRE_CONDITION_CLASSES),
        )
        self.loss_log_vars = nn.ParameterDict({
            "tread": nn.Parameter(torch.zeros(())),
            "health": nn.Parameter(torch.zeros(())),
            "life": nn.Parameter(torch.zeros(())),
            "wear": nn.Parameter(torch.zeros(())),
            "condition": nn.Parameter(torch.zeros(())),
        })
        self._transformer_dim = transformer_dim

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = trainable
        if self.transformer is not None:
            for p in self.transformer.parameters():
                p.requires_grad = trainable

    def unfreeze_last_blocks(self) -> None:
        self.set_encoder_trainable(False)
        if hasattr(self.cnn, "unfreeze_last_blocks"):
            self.cnn.unfreeze_last_blocks()
        else:
            for p in list(self.cnn.parameters())[-20:]:
                p.requires_grad = True
        if self.transformer is not None and hasattr(self.transformer, "unfreeze_last_blocks"):
            self.transformer.unfreeze_last_blocks()
        elif self.transformer is not None:
            for p in list(self.transformer.parameters())[-20:]:
                p.requires_grad = True

    def forward(self, inputs: dict[str, torch.Tensor] | torch.Tensor, tread_sequence: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if isinstance(inputs, dict):
            image = inputs["image"]
            sequence = inputs["tread_sequence"]
        else:
            image = inputs
            if tread_sequence is None:
                raise ValueError("tread_sequence required when inputs is a tensor")
            sequence = tread_sequence
        cnn_features = self.cnn(image)
        transformer_features = self.transformer(image) if self.transformer is not None else None
        rnn_features = self.rnn(sequence)
        if transformer_features is not None:
            vision_fused = torch.cat([cnn_features, transformer_features], dim=1)
        else:
            vision_fused = cnn_features
        fused = self.fusion(cnn_features, transformer_features, rnn_features)
        return {
            "tread_depths": self.tread_head(vision_fused),
            "health_score": self.health_head(fused),
            "remaining_life": self.life_head(fused),
            "wear_pattern": self.wear_head(fused),
            "condition": self.condition_head(fused),
        }


class SmartModelFactory:
    @staticmethod
    def from_dataset_size(dataset_size: int, pretrained: bool = True) -> SmartTireModel:
        config = select_models(dataset_size)
        category = config["dataset_category"]
        logger.info("Dataset category: %s (%d samples)", category, dataset_size)
        logger.info("  CNN: %s, Transformer: %s, RNN: %s, Fusion: %s",
                    config["cnn"], config["transformer"] or "None", config["rnn"], config["fusion"])
        return SmartTireModel(
            cnn_name=config["cnn"],
            transformer_name=config["transformer"],
            rnn_name=config["rnn"],
            fusion_name=config["fusion"],
            pretrained=pretrained,
            cnn_feature_dim=config["cnn_feature_dim"],
            rnn_output_dim=config["rnn_output_dim"],
            fusion_output_dim=config["fusion_output_dim"],
        )

    @staticmethod
    def from_config(config: dict[str, Any], pretrained: bool = True) -> SmartTireModel:
        return SmartTireModel(
            cnn_name=config.get("cnn", "EfficientNetV2-B0"),
            transformer_name=config.get("transformer", None),
            rnn_name=config.get("rnn", "BiLSTM"),
            fusion_name=config.get("fusion", "Deep Dense Fusion"),
            pretrained=pretrained,
            cnn_feature_dim=config.get("cnn_feature_dim", 512),
            rnn_output_dim=config.get("rnn_output_dim", 256),
            fusion_output_dim=config.get("fusion_output_dim", 512),
        )
