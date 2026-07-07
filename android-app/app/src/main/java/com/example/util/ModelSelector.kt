package com.example.util

import android.content.Context
import android.util.Log

object ModelSelector {
    private const val TAG = "ModelSelector"

    data class SelectionResult(
        val modelType: ModelType,
        val recommendation: ModelRecommendation,
        val specs: DeviceSpecs,
        val useGpu: Boolean,
        val useNnapi: Boolean,
        val numThreads: Int,
    ) {
        fun toLogString(): String = buildString {
            appendLine("=== Model Selection Result ===")
            appendLine("Device: ${specs.totalRamMb}MB RAM, SDK ${specs.androidSdk}, " +
                "GPU=${specs.hasGpu}, NNAPI=${specs.hasNnapi}, Cores=${specs.numCores}")
            appendLine("Tier: ${recommendation.deviceTier} / ${recommendation.datasetTier}")
            appendLine("Acceleration: GPU=$useGpu NNAPI=$useNnapi Threads=$numThreads")
            appendLine("Architecture:")
            appendLine("  CNN: ${recommendation.cnnModel.name} " +
                "(${recommendation.cnnModel.paramsM}M params, ${recommendation.cnnModel.sizeMb}MB)")
            if (recommendation.transformerModel != null) {
                appendLine("  Transformer: ${recommendation.transformerModel.name} " +
                    "(${recommendation.transformerModel.paramsM}M params, ${recommendation.transformerModel.sizeMb}MB)")
            }
            appendLine("  RNN: ${recommendation.rnnModel.name} " +
                "(${recommendation.rnnModel.paramsM}M params, ${recommendation.rnnModel.sizeMb}MB)")
            appendLine("  Fusion: ${recommendation.fusionModel.name} " +
                "(${recommendation.fusionModel.paramsM}M params, ${recommendation.fusionModel.sizeMb}MB)")
            appendLine("Runtime: ${modelType.filename}")
            appendLine("Reasoning: ${recommendation.reasoning}")
        }
    }

    enum class ModelType(val filename: String, val description: String, val quality: Int) {
        INT8("model_int8.tflite", "INT8 quantized (fastest, smallest)", 1),
        FP16("model_fp16.tflite", "FP16 quantized (balanced)", 2),
        DYNAMIC_RANGE("model_dynamic.tflite", "Dynamic range quantized", 3),
        FP32("model_latest.tflite", "FP32 (most accurate, largest)", 4),
        ONNX("model.onnx", "ONNX (ONNX Runtime)", 5),
    }

    fun select(
        context: Context,
        trainCount: Int = 0,
        preferAccuracy: Boolean = false,
        preferSpeed: Boolean = false,
    ): SelectionResult {
        val specs = SystemSpecProvider.getDeviceSpecs(context)
        val recommendation = AndroidModelRegistry.recommend(specs, trainCount)

        val availableModels = ModelType.entries.filter {
            try {
                context.assets.open(it.filename).use { true }
            } catch (e: Exception) {
                false
            }
        }

        val useGpu = SystemSpecProvider.isGpuAccelerationSupported(specs) &&
            specs.totalRamMb >= 4096
        val useNnapi = SystemSpecProvider.isNnapiSupported(specs) &&
            !useGpu

        val numThreads = when {
            specs.numCores >= 8 -> 6
            specs.numCores >= 6 -> 4
            specs.numCores >= 4 -> 3
            else -> 2
        }

        val modelType = selectModelType(availableModels, specs, preferAccuracy, preferSpeed)

        Log.d(TAG, recommendation.toString()) // recommendation is ModelRecommendation which doesn't have toLogString()

        return SelectionResult(
            modelType = modelType,
            recommendation = recommendation,
            specs = specs,
            useGpu = useGpu,
            useNnapi = useNnapi,
            numThreads = numThreads,
        )
    }

    private fun selectModelType(
        available: List<ModelType>,
        specs: DeviceSpecs,
        preferAccuracy: Boolean,
        preferSpeed: Boolean,
    ): ModelType {
        val hasLowRam = specs.isLowRam || specs.totalRamMb < 3072
        val hasLimitedStorage = specs.availableStorageMb < 500

        val filtered = available.toMutableList()

        if (available.any { it == ModelType.ONNX }) {
            return ModelType.ONNX
        }

        if (hasLowRam || hasLimitedStorage || preferSpeed) {
            filtered.remove(ModelType.FP32)
        }

        if (preferAccuracy) {
            filtered.remove(ModelType.INT8)
        }

        if (filtered.isEmpty()) {
            return available.maxByOrNull { it.quality } ?: ModelType.FP32
        }

        return when {
            preferSpeed -> {
                filtered.minByOrNull { it.quality } ?: filtered.first()
            }
            preferAccuracy -> {
                filtered.maxByOrNull { it.quality } ?: filtered.last()
            }
            hasLowRam || hasLimitedStorage -> {
                val candidate = filtered.firstOrNull { it == ModelType.INT8 }
                candidate ?: filtered.first()
            }
            else -> {
                val candidate = filtered.firstOrNull { it == ModelType.FP16 }
                candidate ?: filtered.first()
            }
        }
    }

    @JvmStatic
    fun selectModel(context: Context): ModelType {
        val result = select(context, preferAccuracy = true)
        return result.modelType
    }

    fun getInferenceConfig(result: SelectionResult): InferenceConfig {
        val isOnnx = result.modelType == ModelType.ONNX
        return InferenceConfig(
            modelFileName = result.modelType.filename,
            numThreads = result.numThreads,
            useGpu = result.useGpu || isOnnx,
            useNnapi = result.useNnapi,
            inputSize = getInputSize(result.recommendation.cnnModel.name),
            useOnnx = isOnnx,
        )
    }

    private fun getInputSize(modelName: String): Int {
        return when {
            modelName.contains("EfficientNet") -> 224
            modelName == "ConvNeXt-Tiny" -> 224
            modelName.contains("MobileNet") -> 224
            modelName.contains("Squeeze") || modelName.contains("Shuffle") -> 224
            else -> 224
        }
    }
}

data class InferenceConfig(
    val modelFileName: String,
    val numThreads: Int,
    val useGpu: Boolean,
    val useNnapi: Boolean,
    val inputSize: Int = 224,
    val useOnnx: Boolean = false,
)
