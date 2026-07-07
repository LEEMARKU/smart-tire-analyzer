package com.example.util

enum class ModelFamily { CNN, TRANSFORMER, RNN, FUSION }

data class AndroidModelInfo(
    val name: String,
    val family: ModelFamily,
    val paramsM: Double,
    val sizeMb: Double,
    val ramMb: Int,
    val latency: String,
    val minRam: Int,
    val minAndroid: Int,
    val featureDim: Int,
    val isVision: Boolean = true,
    val requiresGpu: Boolean = false,
    val tfliteName: String = name.lowercase()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "") + ".tflite",
)

object AndroidModelRegistry {

    private val cnnModels = listOf(
        AndroidModelInfo("MobileNetV1", ModelFamily.CNN, 4.2, 16.0, 200, "fast", 2048, 24, 1024),
        AndroidModelInfo("MobileNetV2", ModelFamily.CNN, 3.5, 14.0, 180, "fast", 2048, 24, 1280),
        AndroidModelInfo("MobileNetV3", ModelFamily.CNN, 5.4, 22.0, 250, "fast", 3072, 24, 1280),
        AndroidModelInfo("EfficientNet-B0", ModelFamily.CNN, 5.3, 20.0, 300, "medium", 3072, 24, 1280),
        AndroidModelInfo("EfficientNet-Lite0", ModelFamily.CNN, 4.7, 18.0, 250, "fast", 2048, 24, 1280),
        AndroidModelInfo("EfficientNetV2-B0", ModelFamily.CNN, 7.1, 28.0, 350, "medium", 4096, 26, 1280),
        AndroidModelInfo("SqueezeNet", ModelFamily.CNN, 1.2, 5.0, 150, "very_fast", 1024, 21, 512),
        AndroidModelInfo("ShuffleNetV2", ModelFamily.CNN, 2.3, 9.0, 160, "very_fast", 1024, 21, 768),
        AndroidModelInfo("NASNet-Mobile", ModelFamily.CNN, 5.7, 23.0, 280, "medium", 3072, 24, 768),
        AndroidModelInfo("GhostNet", ModelFamily.CNN, 5.2, 20.0, 220, "fast", 2048, 24, 960),
        AndroidModelInfo("MnasNet", ModelFamily.CNN, 4.4, 17.0, 200, "fast", 2048, 24, 1280),
        AndroidModelInfo("ConvNeXt-Tiny", ModelFamily.CNN, 28.0, 110.0, 600, "slow", 6144, 28, 768),
    )

    private val transformerModels = listOf(
        AndroidModelInfo("MobileViT", ModelFamily.TRANSFORMER, 5.6, 22.0, 350, "medium", 3072, 26, 384),
        AndroidModelInfo("LeViT", ModelFamily.TRANSFORMER, 7.8, 31.0, 400, "medium", 4096, 26, 384),
        AndroidModelInfo("TinyViT", ModelFamily.TRANSFORMER, 5.0, 20.0, 300, "medium", 3072, 26, 256),
        AndroidModelInfo("DeiT-Tiny", ModelFamily.TRANSFORMER, 5.7, 23.0, 320, "medium", 3072, 26, 256),
        AndroidModelInfo("EfficientFormer", ModelFamily.TRANSFORMER, 6.2, 25.0, 350, "medium", 4096, 26, 512),
        AndroidModelInfo("EdgeNeXt", ModelFamily.TRANSFORMER, 5.0, 20.0, 300, "medium", 3072, 26, 384),
        AndroidModelInfo("SwiftFormer", ModelFamily.TRANSFORMER, 3.5, 14.0, 250, "fast", 2048, 24, 384),
        AndroidModelInfo("PiT", ModelFamily.TRANSFORMER, 4.9, 19.0, 280, "medium", 3072, 24, 256),
    )

    private val rnnModels = listOf(
        AndroidModelInfo("LSTM", ModelFamily.RNN, 0.5, 2.0, 80, "very_fast", 512, 21, 256, false),
        AndroidModelInfo("BiLSTM", ModelFamily.RNN, 1.0, 4.0, 120, "very_fast", 1024, 21, 256, false),
        AndroidModelInfo("GRU", ModelFamily.RNN, 0.4, 2.0, 80, "very_fast", 512, 21, 256, false),
        AndroidModelInfo("BiGRU", ModelFamily.RNN, 0.8, 3.0, 100, "very_fast", 1024, 21, 256, false),
        AndroidModelInfo("Simple RNN", ModelFamily.RNN, 0.2, 1.0, 50, "very_fast", 512, 21, 128, false),
        AndroidModelInfo("TCN", ModelFamily.RNN, 0.8, 3.0, 100, "very_fast", 1024, 21, 256, false),
    )

    private val fusionModels = listOf(
        AndroidModelInfo("MLP", ModelFamily.FUSION, 2.1, 8.0, 150, "fast", 1024, 21, 512, false),
        AndroidModelInfo("Deep Dense Fusion", ModelFamily.FUSION, 3.5, 14.0, 200, "fast", 2048, 24, 512, false),
        AndroidModelInfo("Early Fusion", ModelFamily.FUSION, 0.5, 2.0, 80, "very_fast", 512, 21, 512, false),
        AndroidModelInfo("Late Fusion", ModelFamily.FUSION, 2.0, 8.0, 160, "fast", 1024, 24, 512, false),
        AndroidModelInfo("Hybrid Fusion", ModelFamily.FUSION, 2.8, 11.0, 180, "fast", 2048, 24, 512, false),
        AndroidModelInfo("Cross-Modal Attention", ModelFamily.FUSION, 4.2, 17.0, 300, "medium", 3072, 26, 512, false),
        AndroidModelInfo("Attention-Based Fusion", ModelFamily.FUSION, 2.5, 10.0, 180, "fast", 2048, 24, 512, false),
        AndroidModelInfo("Gated Fusion", ModelFamily.FUSION, 2.2, 9.0, 170, "fast", 2048, 24, 512, false),
        AndroidModelInfo("Residual Fusion", ModelFamily.FUSION, 2.0, 8.0, 150, "fast", 2048, 24, 512, false),
        AndroidModelInfo("Adaptive Fusion", ModelFamily.FUSION, 3.0, 12.0, 200, "fast", 2048, 24, 512, false),
    )

    private val allModels: Map<String, AndroidModelInfo> by lazy {
        (cnnModels + transformerModels + rnnModels + fusionModels).associateBy { it.name }
    }

    private val cnnByTier = mapOf(
        "low" to listOf("SqueezeNet", "ShuffleNetV2"),
        "low_medium" to listOf("MobileNetV1", "MobileNetV2", "EfficientNet-Lite0", "MnasNet"),
        "medium" to listOf("MobileNetV3", "GhostNet", "NASNet-Mobile", "EfficientNet-B0"),
        "high_medium" to listOf("MobileNetV3", "GhostNet", "EfficientNet-B0", "EfficientNetV2-B0"),
        "high" to listOf("EfficientNetV2-B0", "ConvNeXt-Tiny"),
    )

    private val transformerByTier = mapOf(
        "low" to emptyList(),
        "low_medium" to listOf("SwiftFormer"),
        "medium" to listOf("MobileViT", "TinyViT", "DeiT-Tiny", "EdgeNeXt", "PiT"),
        "high_medium" to listOf("MobileViT", "EfficientFormer", "LeViT"),
        "high" to listOf("EfficientFormer", "LeViT"),
    )

    private val rnnByTier = mapOf(
        "low" to listOf("Simple RNN", "LSTM", "GRU"),
        "low_medium" to listOf("BiLSTM", "BiGRU", "TCN"),
        "medium" to listOf("BiLSTM", "BiGRU", "TCN"),
        "high_medium" to listOf("BiLSTM", "BiGRU", "TCN"),
        "high" to listOf("BiLSTM", "BiGRU", "TCN"),
    )

    private val fusionByTier = mapOf(
        "low" to listOf("Early Fusion", "MLP"),
        "low_medium" to listOf("Late Fusion", "Residual Fusion", "Gated Fusion"),
        "medium" to listOf("Hybrid Fusion", "Attention-Based Fusion", "Deep Dense Fusion", "Adaptive Fusion"),
        "high_medium" to listOf("Hybrid Fusion", "Adaptive Fusion", "Cross-Modal Attention"),
        "high" to listOf("Cross-Modal Attention", "Adaptive Fusion"),
    )

    private val datasetTierThresholds = listOf(
        "tiny" to 0,
        "very_small" to 200,
        "small" to 500,
        "medium" to 2000,
        "large" to 5000,
        "very_large" to 15000,
    )

    fun getAll(): List<AndroidModelInfo> = allModels.values.toList()

    fun getByFamily(family: ModelFamily): List<AndroidModelInfo> = when (family) {
        ModelFamily.CNN -> cnnModels
        ModelFamily.TRANSFORMER -> transformerModels
        ModelFamily.RNN -> rnnModels
        ModelFamily.FUSION -> fusionModels
    }

    fun get(name: String): AndroidModelInfo? = allModels[name]

    fun getDatasetTier(trainCount: Int): String {
        var tier = "very_large"
        for ((name, threshold) in datasetTierThresholds) {
            if (trainCount >= threshold) tier = name
        }
        return tier
    }

    fun getDeviceTier(specs: DeviceSpecs): String {
        return when {
            specs.totalRamMb >= 6144 && specs.androidSdk >= 28 && specs.hasGpu -> "high"
            specs.totalRamMb >= 4096 && specs.androidSdk >= 28 -> "high_medium"
            specs.totalRamMb >= 3072 && specs.androidSdk >= 26 -> "medium"
            specs.totalRamMb >= 2048 && specs.androidSdk >= 24 -> "low_medium"
            else -> "low"
        }
    }

    private fun pickBest(
        candidates: List<String>,
        datasetTier: String,
        family: ModelFamily,
    ): AndroidModelInfo {
        val available = candidates.mapNotNull { allModels[it] }
        if (available.isEmpty()) {
            return getByFamily(family).first()
        }

        val tierScores = mapOf(
            "tiny" to 0, "very_small" to 1, "small" to 2,
            "medium" to 3, "large" to 4, "very_large" to 5,
        )
        val score = tierScores[datasetTier] ?: 3

        val sorted = when {
            score <= 1 -> available.sortedBy { it.paramsM }
            score <= 3 -> available.sortedBy { it.paramsM }
            else -> available.sortedByDescending { it.featureDim }
        }

        val idx = minOf(score, sorted.size - 1).coerceAtLeast(0)
        return sorted[idx]
    }

    fun recommend(specs: DeviceSpecs, trainCount: Int): ModelRecommendation {
        val deviceTier = getDeviceTier(specs)
        val datasetTier = getDatasetTier(trainCount)

        val cnnCandidates = cnnByTier[deviceTier] ?: cnnByTier["low_medium"]!!
        val transformerCandidates = transformerByTier[deviceTier] ?: emptyList()
        val rnnCandidates = rnnByTier[deviceTier] ?: rnnByTier["low"]!!
        val fusionCandidates = fusionByTier[deviceTier] ?: fusionByTier["low"]!!

        val cnnModel = pickBest(cnnCandidates, datasetTier, ModelFamily.CNN)
        val transformerModel = if (transformerCandidates.isNotEmpty()) {
            pickBest(transformerCandidates, datasetTier, ModelFamily.TRANSFORMER)
        } else null
        val rnnModel = pickBest(rnnCandidates, datasetTier, ModelFamily.RNN)
        val fusionModel = pickBest(fusionCandidates, datasetTier, ModelFamily.FUSION)

        val parts = mutableListOf(
            "Device tier: '$deviceTier' (${specs.totalRamMb}MB RAM, SDK ${specs.androidSdk}).",
            "Dataset tier: '$datasetTier' ($trainCount samples).",
        )
        if (transformerModel == null) {
            parts.add("Transformer disabled - device too constrained.")
        } else {
            parts.add("Device capable of transformer model.")
        }
        parts.add(
            "CNN: ${cnnModel.name}, Transformer: ${transformerModel?.name ?: "None"}, " +
                "RNN: ${rnnModel.name}, Fusion: ${fusionModel.name}."
        )

        return ModelRecommendation(
            cnnModel = cnnModel,
            transformerModel = transformerModel,
            rnnModel = rnnModel,
            fusionModel = fusionModel,
            deviceTier = deviceTier,
            datasetTier = datasetTier,
            reasoning = parts.joinToString(" "),
        )
    }
}

data class ModelRecommendation(
    val cnnModel: AndroidModelInfo,
    val transformerModel: AndroidModelInfo?,
    val rnnModel: AndroidModelInfo,
    val fusionModel: AndroidModelInfo,
    val deviceTier: String,
    val datasetTier: String,
    val reasoning: String,
)
