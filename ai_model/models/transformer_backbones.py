from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("transformer_backbones")

TRANSFORMER_REGISTRY: dict[str, dict] = {}


def register_transformer(name: str, feature_dim: int = 512):
    def decorator(cls):
        TRANSFORMER_REGISTRY[name] = {"class": cls, "feature_dim": feature_dim}
        return cls
    return decorator


class BaseTransformerBackbone(nn.Module):
    feature_dim: int = 512

    def __init__(self, pretrained: bool = True, feature_dim: int | None = None):
        super().__init__()
        if feature_dim is not None:
            self.feature_dim = feature_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unfreeze_last_blocks(self, num_layers: int = 4) -> None:
        pass


def _project_features(in_features: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, out_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(out_dim),
        nn.Dropout(0.2),
    )


class _TimmTransformerBackbone(BaseTransformerBackbone):
    def __init__(self, timm_name: str, pretrained: bool, feature_dim: int, num_features: int | None = None):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.timm_name = timm_name
        self.encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        in_features = num_features or getattr(self.encoder, "num_features", 768)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        return self.projector(features)

    def unfreeze_last_blocks(self, num_layers: int = 4) -> None:
        if hasattr(self.encoder, "blocks"):
            for block in self.encoder.blocks[-num_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif hasattr(self.encoder, "layers"):
            for layer in self.encoder.layers[-num_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True


_TRANSFORMER_MODELS = [
    ("ViT", "vit_base_patch16_224", 768),
    ("DeiT", "deit_base_patch16_224", 768),
    ("Swin Transformer", "swin_tiny_patch4_window7_224", 768),
    ("PVT", "pvt_v2_b0", 256),
    ("BEiT", "beit_base_patch16_224", 768),
    ("MaxViT", "maxvit_tiny_rw_224", 512),
    ("CaiT", "cait_xxs24_224", 384),
    ("LeViT", "levit_128", 384),
    ("PiT", "pit_ti_224", 256),
]


class _CustomTransformerBackbone(BaseTransformerBackbone):
    def __init__(self, name: str, pretrained: bool, feature_dim: int, encoder_fn=None):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        self.encoder_fn = encoder_fn
        self.projector = _project_features(feature_dim, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("CvT", feature_dim=384)
class CvTBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 384):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("convit_tiny", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 768)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("T2T-ViT", feature_dim=512)
class T2TViTBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("deit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 192)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))


@register_transformer("RegionViT", feature_dim=512)
class RegionViTBackbone(BaseTransformerBackbone):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__(pretrained=pretrained, feature_dim=feature_dim)
        import timm
        self.encoder = timm.create_model("crossvit_9_240", pretrained=pretrained, num_classes=0)
        in_features = getattr(self.encoder, "num_features", 512)
        self.projector = _project_features(in_features, feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(image))

def _make_transformer_variant(timm_name: str, default_dim: int, display_name: str):
    class _DynamicTransformerVariant(_TimmTransformerBackbone):
        def __init__(self, pretrained: bool = True, feature_dim: int | None = None):
            super().__init__(timm_name, pretrained, feature_dim or default_dim)
        def __repr__(self):
            return f"{display_name}Backbone(dim={self.feature_dim})"
    return _DynamicTransformerVariant


for _name, _timm_name, _feat_dim in _TRANSFORMER_MODELS:
    cls = _make_transformer_variant(_timm_name, _feat_dim, _name)
    register_transformer(_name, feature_dim=_feat_dim)(cls)


class TransformerBackboneFactory:
    @staticmethod
    def list_models() -> list[str]:
        return sorted(TRANSFORMER_REGISTRY.keys())

    @staticmethod
    def get_info(model_name: str) -> dict:
        normalized = _normalize_name(model_name)
        if normalized not in TRANSFORMER_REGISTRY:
            raise KeyError(f"Unknown Transformer model: {model_name}. Available: {sorted(TRANSFORMER_REGISTRY.keys())}")
        info = TRANSFORMER_REGISTRY[normalized]
        return {"name": normalized, "feature_dim": info["feature_dim"]}

    @staticmethod
    def create(model_name: str, pretrained: bool = True, feature_dim: int | None = None) -> BaseTransformerBackbone:
        normalized = _normalize_name(model_name)
        if normalized not in TRANSFORMER_REGISTRY:
            raise KeyError(f"Unknown Transformer model: {model_name}. Available: {sorted(TRANSFORMER_REGISTRY.keys())}")
        info = TRANSFORMER_REGISTRY[normalized]
        return info["class"](pretrained=pretrained, feature_dim=feature_dim or info["feature_dim"])


def _normalize_name(name: str) -> str:
    mapping = {
        "vit": "ViT",
        "deit": "DeiT",
        "swin": "Swin Transformer",
        "swin_transformer": "Swin Transformer",
        "swintransformer": "Swin Transformer",
        "cvt": "CvT",
        "pvt": "PVT",
        "t2t_vit": "T2T-ViT",
        "t2tvit": "T2T-ViT",
        "beit": "BEiT",
        "maxvit": "MaxViT",
        "cait": "CaiT",
        "levit": "LeViT",
        "pit": "PiT",
        "regionvit": "RegionViT",
        "region_vit": "RegionViT",
    }
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, name)
