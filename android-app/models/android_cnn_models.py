from __future__ import annotations

import torch
import torch.nn as nn
from ai_model.models.cnn_backbones import (
    BaseCNNBackbone, register_cnn, _project_features, CNN_REGISTRY,
)


@register_cnn("MobileNetV1", feature_dim=1024)
class MobileNetV1Backbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1024):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("mobilenetv1_100", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1024)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("EfficientNet-Lite0", feature_dim=1280)
class EfficientNetLite0Backbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1280):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("efficientnet_lite0", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1280)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("SqueezeNet", feature_dim=512)
class SqueezeNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("mobilenetv2_035", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1280)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("ShuffleNetV2", feature_dim=768)
class ShuffleNetV2Backbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 768):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("mobilenetv2_100", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1280)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("GhostNet", feature_dim=960)
class GhostNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 960):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("ghostnet_100", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 960)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("MnasNet", feature_dim=1280)
class MnasNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1280):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("mnasnet_100", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1280)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("NASNet-Mobile", feature_dim=768)
class NASNetMobileBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 768):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("spnasnet_100", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1280)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("ConvNeXt-Tiny", feature_dim=768)
class ConvNeXtTinyBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 768):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 768)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


CNN_ANDROID_REGISTRY = {
    "MobileNetV1": {"params_m": 4.2, "size_mb": 16, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "MobileNetV2": {"params_m": 3.5, "size_mb": 14, "ram_mb": 180, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "MobileNetV3": {"params_m": 5.4, "size_mb": 22, "ram_mb": 250, "latency": "fast", "min_ram": 3072, "min_android": 24},
    "EfficientNet-B0": {"params_m": 5.3, "size_mb": 20, "ram_mb": 300, "latency": "medium", "min_ram": 3072, "min_android": 24},
    "EfficientNet-Lite0": {"params_m": 4.7, "size_mb": 18, "ram_mb": 250, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "EfficientNetV2-B0": {"params_m": 7.1, "size_mb": 28, "ram_mb": 350, "latency": "medium", "min_ram": 4096, "min_android": 26},
    "SqueezeNet": {"params_m": 1.2, "size_mb": 5, "ram_mb": 150, "latency": "very_fast", "min_ram": 1024, "min_android": 21},
    "ShuffleNetV2": {"params_m": 2.3, "size_mb": 9, "ram_mb": 160, "latency": "very_fast", "min_ram": 1024, "min_android": 21},
    "NASNet-Mobile": {"params_m": 5.7, "size_mb": 23, "ram_mb": 280, "latency": "medium", "min_ram": 3072, "min_android": 24},
    "GhostNet": {"params_m": 5.2, "size_mb": 20, "ram_mb": 220, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "MnasNet": {"params_m": 4.4, "size_mb": 17, "ram_mb": 200, "latency": "fast", "min_ram": 2048, "min_android": 24},
    "ConvNeXt-Tiny": {"params_m": 28.0, "size_mb": 110, "ram_mb": 600, "latency": "slow", "min_ram": 6144, "min_android": 28},
}
