# Register Android-optimized mobile models (MobileNetV1, ShuffleNetV2, MobileViT, etc.)
try:
    import android_app.models.android_cnn_models  # noqa: F401
    import android_app.models.android_transformer_models  # noqa: F401
except ImportError:
    pass

from ai_model.models.cnn_backbones import CNNBackboneFactory
from ai_model.models.transformer_backbones import TransformerBackboneFactory
from ai_model.models.rnn_encoders import RNNEncoderFactory
from ai_model.models.fusion_networks import FusionNetworkFactory
from ai_model.models.model_factory import (
    SmartModelFactory,
    build_model_component,
    list_available_models,
    list_cnn_models,
    list_transformer_models,
    list_rnn_models,
    list_fusion_models,
)

__all__ = [
    "CNNBackboneFactory",
    "TransformerBackboneFactory",
    "RNNEncoderFactory",
    "FusionNetworkFactory",
    "SmartModelFactory",
    "build_model_component",
    "list_available_models",
    "list_cnn_models",
    "list_transformer_models",
    "list_rnn_models",
    "list_fusion_models",
]
