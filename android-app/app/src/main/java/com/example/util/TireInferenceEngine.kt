package com.example.util

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

data class TireInferenceResult(
    val condition: String,
    val conditionConfidence: Float,
    val health: Float,
    val remainingLife: Float,
    val inferenceTimeMs: Long = 0L,
)

class TireInferenceEngine(private val context: Context) {

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isLoaded = false
    private var currentConfig: InferenceConfig? = null

    private val labels = arrayOf("safe", "moderate", "replace")

    fun loadWithSelection(trainCount: Int = 0, preferAccuracy: Boolean = false) {
        val result = ModelSelector.select(context, trainCount, preferAccuracy)
        val config = ModelSelector.getInferenceConfig(result)
        loadWithConfig(config)
    }

    fun loadWithConfig(config: InferenceConfig) {
        if (isLoaded && currentConfig == config) return
        close()

        val modelBytes = context.assets.open(config.modelFileName).readBytes()
        ortEnv = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            // setNumThreads is not available in session options this way in some ORT versions
            // or might be setIntraOpNumThreads
            setIntraOpNumThreads(config.numThreads)
            try {
                addNnapi()
            } catch (_: Exception) {}
        }
        ortSession = ortEnv!!.createSession(modelBytes, options)
        currentConfig = config
        isLoaded = true
    }

    fun loadModel(modelName: String = "model.onnx") {
        if (isLoaded) return
        val config = InferenceConfig(
            modelFileName = modelName,
            numThreads = 4,
            useGpu = false,
            useNnapi = false,
        )
        loadWithConfig(config)
    }

    fun infer(bitmap: Bitmap): TireInferenceResult {
        val env = ortEnv ?: throw IllegalStateException("ONNX Runtime not initialized")
        val session = ortSession ?: throw IllegalStateException("Model not loaded")
        val inputSize = currentConfig?.inputSize ?: 224

        val startTime = System.currentTimeMillis()

        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val mean = 127.5f
        val std = 127.5f

        val imageData = FloatArray(3 * inputSize * inputSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            imageData[i] = (r - mean) / std                              // R channel
            imageData[inputSize * inputSize + i] = (g - mean) / std      // G channel
            imageData[2 * inputSize * inputSize + i] = (b - mean) / std  // B channel
        }

        val imageShape = longArrayOf(1L, 3L, inputSize.toLong(), inputSize.toLong())
        val imageTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(imageData), imageShape)

        val treadData = FloatArray(4 * 7)
        val treadShape = longArrayOf(1L, 4L, 7L)
        val treadTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(treadData), treadShape)

        val inputs = mapOf(
            "image" to imageTensor,
            "tread_sequence" to treadTensor,
        )

        val results = session.run(inputs)

        val conditionTensor = results.get("condition").get() as OnnxTensor
        val conditionBuffer = FloatBuffer.allocate(3)
        conditionTensor.floatBuffer.rewind()
        conditionTensor.floatBuffer.get(conditionBuffer.array(), 0, 3)

        val conditionIdx = (0 until 3).maxByOrNull { conditionBuffer.array()[it] } ?: 0

        val healthTensor = results.get("health_score").get() as OnnxTensor
        healthTensor.floatBuffer.rewind()
        val healthScore = healthTensor.floatBuffer.get()

        val lifeTensor = results.get("remaining_life").get() as OnnxTensor
        lifeTensor.floatBuffer.rewind()
        val remainingLife = lifeTensor.floatBuffer.get()

        imageTensor.close()
        treadTensor.close()
        results.close()

        val inferenceTime = System.currentTimeMillis() - startTime

        return TireInferenceResult(
            condition = labels[conditionIdx],
            conditionConfidence = conditionBuffer.array()[conditionIdx],
            health = healthScore,
            remainingLife = remainingLife,
            inferenceTimeMs = inferenceTime,
        )
    }

    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val bitmap = when (imageProxy.format) {
            ImageFormat.JPEG -> {
                val buffer = imageProxy.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
            ImageFormat.YUV_420_888, ImageFormat.NV21 -> {
                val yBuffer = imageProxy.planes[0].buffer
                val uBuffer = imageProxy.planes[1].buffer
                val vBuffer = imageProxy.planes[2].buffer

                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()

                val nv21 = ByteArray(ySize + uSize + vSize)
                yBuffer.get(nv21, 0, ySize)
                vBuffer.get(nv21, ySize, vSize)
                uBuffer.get(nv21, ySize + vSize, uSize)

                val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
                val jpegBytes = out.toByteArray()
                BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            }
            else -> {
                val buffer = imageProxy.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            }
        }

        if (bitmap == null) return null

        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    fun getModelInfo(): String {
        val config = currentConfig
        if (config == null) return "No model loaded"
        return buildString {
            appendLine("Model: ${config.modelFileName}")
            appendLine("Threads: ${config.numThreads}")
            appendLine("GPU: ${config.useGpu}")
            appendLine("NNAPI: ${config.useNnapi}")
            appendLine("Input size: ${config.inputSize}x${config.inputSize}")
        }
    }

    fun close() {
        ortSession?.close()
        ortSession = null
        isLoaded = false
        currentConfig = null
    }

    fun isModelLoaded(): Boolean = isLoaded
}
