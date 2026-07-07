package com.example.ui.screens

import android.Manifest
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.ui.geometry.Offset
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.ui.components.ScanOverlay
import com.example.viewmodel.TireTwinViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    viewModel: TireTwinViewModel,
    onNavigateBack: () -> Unit,
    onNavigateToResults: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    var hasCameraPermission by remember { mutableStateOf(false) }
    var isScanning by remember { mutableStateOf(false) }
    var scanProgress by remember { mutableStateOf(0f) }
    var activeLogText by remember { mutableStateOf("Ready to initiate scan. Position tread.") }
    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var inferenceError by remember { mutableStateOf<String?>(null) }

    // Request permissions launcher
    val requestPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        // Trigger initial request
        requestPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    // Scanner configuration

    suspend fun runOnnxInference(bitmap: android.graphics.Bitmap) {
        viewModel.runTireInference(bitmap)

        activeLogText = "Synthesizing results..."
        scanProgress = 0.9f
        delay(300)

        val title = "Tire Scan (${System.currentTimeMillis() % 1000})"
        viewModel.saveScanToHistory(title) {
            isScanning = false
            scanProgress = 1f
            onNavigateToResults()
        }
    }

    fun startProfileScanning() {
        if (isScanning) return
        inferenceError = null
        isScanning = true
        scanProgress = 0f

        val capture = imageCapture
        if (capture == null) {
            inferenceError = "Camera not initialized"
            isScanning = false
            return
        }

        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: androidx.camera.core.ImageProxy) {
                    coroutineScope.launch {
                        activeLogText = "Processing image..."
                        scanProgress = 0.3f

                        val bitmap = viewModel.tireInferenceEngine.imageProxyToBitmap(image)
                        image.close()

                        if (bitmap == null) {
                            inferenceError = "Failed to decode camera frame"
                            isScanning = false
                            return@launch
                        }

                        activeLogText = "Running ONNX inference..."
                        scanProgress = 0.6f

                        runOnnxInference(bitmap)
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    inferenceError = "Capture failed: ${exception.message}"
                    isScanning = false
                }
            }
        )
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Viewport: Render CameraX is granted, else beautiful visual placeholder
        if (hasCameraPermission) {
            AndroidView(
                factory = { ctx ->
                    val previewView = PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                    }
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        val capture = ImageCapture.Builder()
                            .setTargetResolution(Size(224, 224))
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .build()
                        imageCapture = capture
                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                capture
                            )
                        } catch (e: Exception) {
                            // Handler
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
        } else {
            // Simulator placeholder
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(Color(0x0A, 0x0D, 0x14)),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CameraAlt,
                        contentDescription = "Simulated Viewport",
                        tint = Color(0x30, 0x36, 0x3D),
                        modifier = Modifier.size(80.dp)
                    )
                    Text(
                        text = "EMULATOR AR VIEWPORT STATUS ACTIVE\n(Optical scanning algorithms simulated)",
                        color = Color(0x8A, 0x9B, 0xA8),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        textAlign = TextAlign.Center,
                        lineHeight = 16.sp
                    )
                }
            }
        }

        // Tactical 3D HUD Reticle Scan Overlay
        ScanOverlay(
            isScanning = isScanning,
            scanProgress = scanProgress,
            modifier = Modifier.fillMaxSize()
        )

        // Top Command Row
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = onNavigateBack,
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                    .size(40.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }

            Text(
                text = "AR TREAD PROFILER",
                color = Color.White,
                fontSize = 13.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            )

            IconButton(
                onClick = { /* Simulated flash trigger */ },
                modifier = Modifier
                    .background(Color.Black.copy(alpha = 0.6f), CircleShape)
                    .size(40.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.FlashOn,
                    contentDescription = "Flash",
                    tint = Color.White
                )
            }
        }

        // Scanning Status Drawer (Bottom Section)
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(20.dp)
                .fillMaxWidth()
                .background(Color.Black.copy(alpha = 0.82f), RoundedCornerShape(14.dp))
                .border(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.3f), RoundedCornerShape(14.dp))
                .padding(20.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (isScanning) {
                    CircularProgressIndicator(
                        progress = { scanProgress },
                        color = MaterialTheme.colorScheme.tertiary,
                        strokeWidth = 3.dp,
                        modifier = Modifier.size(24.dp)
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = "Ready",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(24.dp)
                    )
                }

                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = if (isScanning) "PROFILING TIRES SYSTEM..." else "SCANNER READINESS OK",
                        color = if (isScanning) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.primary,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = activeLogText,
                        color = Color.White,
                        fontSize = 13.sp,
                        maxLines = 1,
                        fontWeight = FontWeight.Medium
                    )
                }

                if (isScanning) {
                    Text(
                        text = "${(scanProgress * 100).toInt()}%",
                        color = MaterialTheme.colorScheme.tertiary,
                        fontSize = 15.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Command trigger button
            Button(
                onClick = ::startProfileScanning,
                enabled = !isScanning,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = Color.Black,
                    disabledContainerColor = Color(0x1F, 0x24, 0x2C)
                ),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(48.dp)
                    .testTag("execute_scan_trigger"),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp)
            ) {
                Text(
                    text = if (isScanning) "CAPTURING DEPTH FIELDS..." else "TRIGGER CAD PROFILE SCAN",
                    fontSize = 13.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    color = if (isScanning) Color.Gray else Color.Black
                )
            }
        }
    }
}
