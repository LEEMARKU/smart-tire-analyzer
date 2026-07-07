from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("model_converter")

ANDROID_ASSETS = Path(__file__).resolve().parents[1] / "app" / "src" / "main" / "assets"


def export_to_tflite(
    model_name: str,
    model_class: str,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    output_dir: str | Path | None = None,
    quantize: str = "fp16",
) -> Path:
    if output_dir is None:
        output_dir = ANDROID_ASSETS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    filename = f"{safe_name}_{quantize}.tflite"
    output_path = output_dir / filename

    script = f"""
import torch
import sys
sys.path.insert(0, r'{Path(__file__).resolve().parents[2]}')

from ai_model.models.cnn_backbones import CNNBackboneFactory
from ai_model.models.transformer_backbones import TransformerBackboneFactory
from ai_model.models.rnn_encoders import RNNEncoderFactory
from ai_model.models.fusion_networks import FusionNetworkFactory

model = {model_class}
model.eval()

dummy_input = torch.randn{input_shape}

import tensorflow as tf

# Convert to TorchScript then to ONNX then to TF
with torch.no_grad():
    traced = torch.jit.trace(model, dummy_input)

# Use torch.onnx -> onnx_tf -> tflite approach
import onnx
import onnx_tf

onnx_path = r'{output_path.with_suffix(".onnx")}'
torch.onnx.export(traced, dummy_input, onnx_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=12)

onnx_model = onnx.load(onnx_path)
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph(r'{output_path.with_suffix("")}_tf')

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(r'{output_path.with_suffix("")}_tf')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

if '{quantize}' == 'int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: [dummy_input.numpy()]
    converter.target_spec.supported_types = [tf.int8]
elif '{quantize}' == 'fp16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()
with open(r'{output_path}', 'wb') as f:
    f.write(tflite_model)

print(f"Exported {{output_path}}")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Export failed: %s", result.stderr)
        raise RuntimeError(f"Model export failed: {result.stderr}")
    logger.info("Export succeeded: %s", result.stdout)
    return output_path


def export_all_cnn(output_dir: str | Path | None = None):
    models = [
        ("MobileNetV1", "CNNFactory.create('MobileNetV1', pretrained=False, feature_dim=1024)", (1, 3, 224, 224)),
        ("MobileNetV2", "CNNFactory.create('MobileNetV2', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("MobileNetV3", "CNNFactory.create('MobileNetV3', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("EfficientNet-B0", "CNNFactory.create('EfficientNet-B0', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("EfficientNet-Lite0", "CNNFactory.create('EfficientNet-Lite0', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("EfficientNetV2-B0", "CNNFactory.create('EfficientNetV2-B0', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("SqueezeNet", "CNNFactory.create('SqueezeNet', pretrained=False, feature_dim=512)", (1, 3, 224, 224)),
        ("ShuffleNetV2", "CNNFactory.create('ShuffleNetV2', pretrained=False, feature_dim=768)", (1, 3, 224, 224)),
        ("NASNet-Mobile", "CNNFactory.create('NASNet-Mobile', pretrained=False, feature_dim=768)", (1, 3, 224, 224)),
        ("GhostNet", "CNNFactory.create('GhostNet', pretrained=False, feature_dim=960)", (1, 3, 224, 224)),
        ("MnasNet", "CNNFactory.create('MnasNet', pretrained=False, feature_dim=1280)", (1, 3, 224, 224)),
        ("ConvNeXt-Tiny", "CNNFactory.create('ConvNeXt-Tiny', pretrained=False, feature_dim=768)", (1, 3, 224, 224)),
    ]
    for name, factory_call, shape in models:
        factory = f"CNNFactory.create('{name}', pretrained=False, feature_dim={shape[1]})"
        export_to_tflite(name, factory, shape, output_dir, quantize="fp16")
        export_to_tflite(name, factory, shape, output_dir, quantize="int8")


def main():
    logging.basicConfig(level=logging.INFO)
    output = ANDROID_ASSETS
    print(f"Exporting models to {output}")
    export_all_cnn(output)


if __name__ == "__main__":
    main()
