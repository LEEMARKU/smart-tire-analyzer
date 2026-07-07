from __future__ import annotations

import torch
import torch.nn as nn
from ai_model.models.transformer_backbones import (
    BaseTransformerBackbone, register_transformer, _project_features, TRANSFORMER_REGISTRY,
)


class _TimmTransformerBackbone(BaseTransformerBackbone):
    def __init__(self, timm_name: str, pretrained: bool, feature_dim: int, num_features: int | None = None):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        in_features = num_features or getattr(self.encoder, "num_features", 768)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("MobileViT", feature_dim=384)
class MobileViTBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 384):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("mobilevit_s", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 640)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("TinyViT", feature_dim=256)
class TinyViTBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("tiny_vit_5m_224", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 320)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("EfficientFormer", feature_dim=512)
class EfficientFormerBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("efficientformer_l1", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 448)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("EdgeNeXt", feature_dim=384)
class EdgeNeXtBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 384):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("edgenext_small", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 512)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("SwiftFormer", feature_dim=384)
class SwiftFormerBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 384):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("swiftformer_xs", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 384)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("DeiT-Tiny", feature_dim=256)
class DeiTTinyBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 192)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


TRANSFORMER_ANDROID_REGISTRY = {
    "MobileViT": {"params_m": 5.6, "size_mb": 22, "ram_mb": 350, "latency": "medium", "min_ram": 3072, "min_android": 26},
    "LeViT": {"params_m": 7.8, "size_mb": 31, "ram_mb": 400, "latency": "medium", "min_ram": 4096, "min_android": 26},
    "TinyViT": {"params_m": 5.0, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 26},
    "DeiT-Tiny": {"params_m": 5.7, "size_mb": 23, "ram_mb": 320, "latency": "medium", "min_ram": 3072, "min_android": 26},
    "EfficientFormer": {"params_m": 6.2, "size_mb": 25, "ram_mb": 350, "latency": "medium", "min_ram": 4096, "min_android": 26},
    "EdgeNeXt": {"params_m": 5.0, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 26},
    "SwiftFormer": {"params_m": 3.5, "size_mb": 14, "ram_mb": 250, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "PiT": {"params_m": 4.9, "size_mb": 19, "ram_mb": 280, "latency": "medium", "min_ram": 3072, "min_android": 24},
}
