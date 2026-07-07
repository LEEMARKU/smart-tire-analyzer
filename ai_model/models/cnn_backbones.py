from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("cnn_backbones")

CNN_REGISTRY: dict[str, dict] = {}


def register_cnn(name: str, feature_dim: int = 512, pretrained: bool = True):
    def decorator(cls):
        CNN_REGISTRY[name] = {"class": cls, "feature_dim": feature_dim}
        return cls
    return decorator


class BaseCNNBackbone(nn.Module):
    feature_dim: int = 512

    def __init__(self, pretrained: bool = True, feature_dim: int | None = None):
        super().__init__()
        if feature_dim is not None:
            self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unfreeze_last_blocks(self) -> None:
        pass


def _project_features(encoder: nn.Module, in_features: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(out_dim),
        nn.Dropout(0.2),
    )


def _timm_backbone(model_name: str, pretrained: bool, feature_dim: int, num_features: int | None = None) -> tuple[nn.Module, nn.Module]:
    import timm
    encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    in_features = num_features or getattr(encoder, "num_features", 1280)
    projector = _project_features(encoder, in_features, feature_dim)
    return encoder, projector


@register_cnn("LeNet", feature_dim=128)
class LeNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 128):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.projector = _project_features(nn.Identity(), 84, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = self.classifier(x)
        return self.projector(x)


@register_cnn("AlexNet", feature_dim=256)
class AlexNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        from torchvision.models import alexnet, AlexNet_Weights
        weights = AlexNet_Weights.DEFAULT if pretrained else None
        self.encoder = alexnet(weights=weights)
        in_features = self.encoder.classifier[6].in_features
        self.encoder.classifier = self.encoder.classifier[:-1]
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)
        return self.projector(x)


@register_cnn("ZFNet", feature_dim=256)
class ZFNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 256):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        in_features = 256 * 6 * 6
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.projector = _project_features(nn.Identity(), 4096, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.projector(x)


class _TimmBackbone(BaseCNNBackbone):
    def __init__(self, timm_name: str, pretrained: bool, feature_dim: int):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = self._resolve_feature_dim(timm_name)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def _resolve_feature_dim(self, timm_name: str) -> int:
        probe = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            try:
                out = self.encoder(probe)
                return out.shape[-1]
            except Exception:
                pass
        classifier = getattr(self.encoder, "classifier", None)
        if classifier is not None:
            if hasattr(classifier, "in_features"):
                return classifier.in_features
            if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
                first = classifier[0]
                if hasattr(first, "in_features"):
                    return first.in_features
        head = getattr(self.encoder, "head", None)
        if head is not None and hasattr(head, "fc") and hasattr(head.fc, "in_features"):
            return head.fc.in_features
        nf = getattr(self.encoder, "num_features", None)
        if nf is not None:
            return nf
        return 1280

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


class _VGGManual(BaseCNNBackbone):
    def __init__(self, variant: str, pretrained: bool, feature_dim: int):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model(variant, pretrained=pretrained)
        self.features = self.encoder.features
        in_features = self._resolve_feat_channels()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def _resolve_feat_channels(self) -> int:
        if hasattr(self.features, "out_channels"):
            return self.features.out_channels
        for m in reversed(list(self.features)):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        head = getattr(self.encoder, "head", None)
        if head is not None and hasattr(head, "fc") and hasattr(head.fc, "in_features"):
            return head.fc.in_features
        classifier = getattr(self.encoder, "classifier", None)
        if classifier is not None:
            if hasattr(classifier, "in_features"):
                return classifier.in_features
            if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
                first = classifier[0]
                if hasattr(first, "in_features"):
                    return first.in_features
        return 512

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.features(image)
        x = self.pool(x).flatten(1)
        return self.projector(x)


def _make_vgg_variant(variant: str, default_dim: int):
    class _DynamicVGGVariant(_VGGManual):
        def __init__(self, pretrained: bool = True, feature_dim: int | None = None):
            super().__init__(variant, pretrained, feature_dim or default_dim)
    return _DynamicVGGVariant


for _variant, _dim in [("vgg16", 512), ("vgg19", 512)]:
    cls = _make_vgg_variant(_variant, _dim)
    register_cnn(_variant.upper(), feature_dim=_dim)(cls)


def _make_timm_variant(timm_name: str, default_dim: int):
    class _DynamicTimmVariant(_TimmBackbone):
        def __init__(self, pretrained: bool = True, feature_dim: int | None = None):
            super().__init__(timm_name, pretrained, feature_dim or default_dim)
    return _DynamicTimmVariant


for _name, _timm_name, _dim in [
    ("ResNet18", "resnet18", 512),
    ("ResNet34", "resnet34", 512),
    ("ResNet50", "resnet50", 2048),
    ("ResNet101", "resnet101", 2048),
    ("DenseNet121", "densenet121", 1024),
    ("DenseNet169", "densenet169", 1664),
    ("DenseNet201", "densenet201", 1920),
    ("MobileNetV2", "mobilenetv2_100", 1280),
    ("MobileNetV3", "mobilenetv3_large_100", 1280),
    ("EfficientNet-B0", "tf_efficientnet_b0", 1280),
    ("EfficientNetV2-B0", "tf_efficientnetv2_b0", 1280),
    ("Xception", "xception", 2048),
    ("ConvNeXt", "convnext_tiny", 768),
]:
    cls = _make_timm_variant(_timm_name, _dim)
    register_cnn(_name, feature_dim=_dim)(cls)


@register_cnn("GoogLeNet", feature_dim=512)
class GoogLeNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        from torchvision.models import googlenet, GoogLeNet_Weights
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        self.encoder = googlenet(weights=weights)
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = _project_features(nn.Identity(), in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)
        return self.projector(x)


@register_cnn("InceptionV3", feature_dim=1024)
class InceptionV3Backbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1024):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("inception_v3", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 2048)
        self.projector = _project_features(self.encoder, in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("Inception-ResNet-v2", feature_dim=1024)
class InceptionResNetV2Backbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1024):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 1536)
        self.projector = _project_features(self.encoder, in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_cnn("NASNet", feature_dim=1024)
class NASNetBackbone(BaseCNNBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 1024):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("nasnetalarge", pretrained=pretrained, num_classes=0, global_pool="avg")
        in_features = getattr(self.encoder, "num_features", 4032)
        self.projector = _project_features(self.encoder, in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


class CNNBackboneFactory:
    @staticmethod
    def list_models() -> list[str]:
        return sorted(CNN_REGISTRY.keys())

    @staticmethod
    def get_info(model_name: str) -> dict:
        normalized = _normalize_name(model_name)
        if normalized not in CNN_REGISTRY:
            raise KeyError(f"Unknown CNN model: {model_name}. Available: {sorted(CNN_REGISTRY.keys())}")
        info = CNN_REGISTRY[normalized]
        return {"name": normalized, "feature_dim": info["feature_dim"]}

    @staticmethod
    def create(model_name: str, pretrained: bool = True, feature_dim: int | None = None) -> BaseCNNBackbone:
        normalized = _normalize_name(model_name)
        if normalized not in CNN_REGISTRY:
            raise KeyError(f"Unknown CNN model: {model_name}. Available: {sorted(CNN_REGISTRY.keys())}")
        info = CNN_REGISTRY[normalized]
        cls = info["class"]
        return cls(pretrained=pretrained, feature_dim=feature_dim or info["feature_dim"])


def _normalize_name(name: str) -> str:
    mapping = {
        "lenet": "LeNet",
        "alexnet": "AlexNet",
        "zfnet": "ZFNet",
        "vgg16": "VGG16",
        "vgg19": "VGG19",
        "googlenet": "GoogLeNet",
        "inceptionv3": "InceptionV3",
        "inception_resnet_v2": "Inception-ResNet-v2",
        "inceptionresnetv2": "Inception-ResNet-v2",
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "resnet50": "ResNet50",
        "resnet101": "ResNet101",
        "densenet121": "DenseNet121",
        "densenet169": "DenseNet169",
        "densenet201": "DenseNet201",
        "mobilenetv2": "MobileNetV2",
        "mobilenetv3": "MobileNetV3",
        "efficientnet_b0": "EfficientNet-B0",
        "efficientnetb0": "EfficientNet-B0",
        "efficientnetv2_b0": "EfficientNetV2-B0",
        "efficientnetv2b0": "EfficientNetV2-B0",
        "xception": "Xception",
        "nasnet": "NASNet",
        "convnext": "ConvNeXt",
    }
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, name)
