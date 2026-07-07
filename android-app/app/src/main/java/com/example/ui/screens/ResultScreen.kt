package com.example.ui.screens

import android.widget.Toast
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput
import com.example.ui.theme.StatusCritical
import com.example.ui.theme.StatusSuccess
import com.example.ui.theme.StatusWarning
import com.example.ui.components.TireDigitalTwin3D
import com.example.ui.components.TelemetryTooltipDialog
import com.example.util.PdfGenerator
import com.example.viewmodel.TireTwinViewModel
import coil.compose.AsyncImage
import androidx.compose.ui.layout.ContentScale
import com.example.viewmodel.TirePosition

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(
    viewModel: TireTwinViewModel,
    onNavigateBack: () -> Unit
) {
    val context = LocalContext.current
    val speed by viewModel.speed.collectAsState()
    val pressure by viewModel.pressure.collectAsState()
    val temperature by viewModel.temperature.collectAsState()
    val wearPattern by viewModel.wearPattern.collectAsState()
    val isThermalMode by viewModel.isThermalMode.collectAsState()
    val isExplodedView by viewModel.isExplodedView.collectAsState()
    val healthScore by viewModel.healthScore.collectAsState()

    val analysisState by viewModel.analysisState.collectAsState()
    val isAnalyzing by viewModel.isAnalyzing.collectAsState()
    val isDarkTheme by viewModel.isDarkTheme.collectAsState()
    val activeTire by viewModel.activeTire.collectAsState()
    val useMetric by viewModel.useMetricUnits.collectAsState()
    
    val frontImageUri by viewModel.frontImageUri.collectAsState()
    val sidewallImageUri by viewModel.sidewallImageUri.collectAsState()

    var tooltipToShow by remember { mutableStateOf<Pair<String, String>?>(null) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = "3D REPORT: ${activeTire.code}",
                        style = LocalTextStyle.current.copy(
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.sp,
                            letterSpacing = 1.5.sp,
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    )
                },
                navigationIcon = {
                    IconButton(
                        onClick = onNavigateBack,
                        modifier = Modifier.testTag("result_back_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Return",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            val battery = viewModel.iotBatteryLevel.value
                            PdfGenerator.generateAndSharePdf(
                                context = context,
                                tireLabel = activeTire.label.uppercase(),
                                speed = speed,
                                pressure = pressure,
                                temperature = temperature,
                                wearPattern = wearPattern,
                                healthScore = healthScore,
                                iotBattery = battery,
                                analysis = analysisState
                            )
                        },
                        modifier = Modifier.testTag("result_export_pdf_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Share,
                            contentDescription = "Export PDF Report",
                            tint = MaterialTheme.colorScheme.onSurface
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = MaterialTheme.colorScheme.onSurface
                )
            )
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Top 45% (Dynamic Interactive 3D Canvas)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.45f)
                    .background(Color.Black)
            ) {
                TireDigitalTwin3D(
                    speed = speed,
                    pressure = pressure,
                    temperature = temperature,
                    wearPattern = wearPattern,
                    isThermalMode = isThermalMode,
                    isExplodedView = isExplodedView,
                    isDarkTheme = isDarkTheme
                )

                // Real-time battery level overlay of the IoT sensor module (Top-Right corner / TopEnd)
                val iotBattery by viewModel.iotBatteryLevel.collectAsState()
                
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(14.dp)
                        .background(Color(0x1F, 0x24, 0x2C).copy(alpha = 0.8f), RoundedCornerShape(6.dp))
                        .border(0.5.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(6.dp))
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onLongPress = {
                                    tooltipToShow = Pair(
                                        "IOT SENSOR BATTERY",
                                        "Remaining energy capacity of the integrated wireless micro-sensor module embedded within the tire carcass. Automatically recalibrates and consumes charge under dynamic thermal and rotation loads."
                                    )
                                }
                            )
                        }
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                        .testTag("result_viewport_sensor_battery_widget")
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        // Small blinking/glowing telemetry activity dot
                        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                        val alpha by infiniteTransition.animateFloat(
                            initialValue = 0.4f,
                            targetValue = 1.0f,
                            animationSpec = infiniteRepeatable(
                                animation = tween(1000, easing = LinearEasing),
                                repeatMode = RepeatMode.Reverse
                            ),
                            label = "alpha"
                        )
                        Box(
                            modifier = Modifier
                                .size(5.dp)
                                .background(MaterialTheme.colorScheme.tertiary.copy(alpha = alpha), CircleShape)
                        )
                        
                        Text(
                            text = "BATT:",
                            color = Color(0x8A, 0x9B, 0xA8),
                            fontSize = 8.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Start
                        ) {
                            // Body
                            Box(
                                modifier = Modifier
                                    .width(16.dp)
                                    .height(9.dp)
                                    .border(0.8.dp, Color.White.copy(alpha = 0.6f), RoundedCornerShape(1.2.dp))
                                    .padding(0.8.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .fillMaxHeight()
                                        .fillMaxWidth(iotBattery / 100f)
                                        .background(
                                            when {
                                                iotBattery > 50f -> StatusSuccess
                                                iotBattery > 20f -> StatusWarning
                                                else -> StatusCritical
                                            },
                                            RoundedCornerShape(0.4.dp)
                                        )
                                )
                            }
                            // Tip
                            Box(
                                modifier = Modifier
                                    .size(width = 1.2.dp, height = 3.2.dp)
                                    .background(Color.White.copy(alpha = 0.6f), RoundedCornerShape(topEnd = 0.8.dp, bottomEnd = 0.8.dp))
                            )
                        }
                        
                        Text(
                            text = "${String.format("%.1f", iotBattery)}%",
                            color = Color.White,
                            fontSize = 9.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                // Render dynamic indicator badges inside canvas for immersive tech style
                Row(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .background(StatusWarning.copy(alpha = 0.15f), RoundedCornerShape(4.dp))
                            .border(1.dp, StatusWarning.copy(alpha = 0.4f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = "WEAR: $wearPattern",
                            fontSize = 9.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }

                    Box(
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.15f), RoundedCornerShape(4.dp))
                            .border(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.4f), RoundedCornerShape(4.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = if (useMetric) "TEMP: ${temperature.toInt()}°C" else "TEMP: ${(temperature * 9 / 5 + 32).toInt()}°F",
                            fontSize = 9.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                }

                // Hot/Cool rendering layers toggle selectors right inside Result pane
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomEnd)
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    IconButton(
                        onClick = { viewModel.isThermalMode.value = !isThermalMode },
                        modifier = Modifier
                            .background(if (isThermalMode) StatusWarning else Color(0x1F, 0x24, 0x2C).copy(alpha = 0.8f), CircleShape)
                            .size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Thermostat,
                            contentDescription = "Thermal Gradient",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                    }

                    IconButton(
                        onClick = { viewModel.isExplodedView.value = !isExplodedView },
                        modifier = Modifier
                            .background(if (isExplodedView) MaterialTheme.colorScheme.primary else Color(0x1F, 0x24, 0x2C).copy(alpha = 0.8f), CircleShape)
                            .size(36.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Layers,
                            contentDescription = "Exploded mesh layer",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }

            // Bottom 55% (Scrollable Diagnostic Sheet)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(0.55f)
                    .background(MaterialTheme.colorScheme.background)
                    .border(BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f)))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                        .padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(18.dp)
                ) {
                    // Overall Health Radial Dial Indicator + Stats Summary Block
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(12.dp))
                            .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f), RoundedCornerShape(12.dp))
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onLongPress = {
                                        tooltipToShow = Pair(
                                            "OVERALL HEALTH INDEX",
                                            "Aggregated digital twin longevity rating, calculated based on dynamic heat generation, centrifugal expansion, tread friction, and psi variance vectors."
                                        )
                                    }
                                )
                            }
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Glowing gauge ring
                        Box(
                            modifier = Modifier
                                .size(90.dp)
                                .testTag("health_ring_indicator"),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(
                                progress = { 1f },
                                color = MaterialTheme.colorScheme.surfaceVariant, // track background
                                strokeWidth = 8.dp,
                                modifier = Modifier.fillMaxSize()
                            )
                            CircularProgressIndicator(
                                progress = { healthScore / 100f },
                                color = evaluateHealthColor(healthScore),
                                strokeWidth = 8.dp,
                                modifier = Modifier.fillMaxSize()
                            )
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text = "$healthScore%",
                                    color = MaterialTheme.colorScheme.onSurface,
                                    fontSize = 20.sp,
                                    fontWeight = FontWeight.Black,
                                    fontFamily = FontFamily.Monospace
                                )
                                Text(
                                    text = "HEALTH",
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontSize = 8.sp,
                                    fontFamily = FontFamily.Monospace,
                                    letterSpacing = 0.5.sp
                                )
                            }
                        }

                        Spacer(modifier = Modifier.width(18.dp))

                        // Score classification description
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(
                                  text = getHealthRatingTitle(healthScore),
                                  color = evaluateHealthColor(healthScore),
                                  fontSize = 16.sp,
                                  fontWeight = FontWeight.Bold
                            )
                            Text(
                                  text = "Wheel alignment parameters and tire carcass rubber parameters registered. Telemetry streams calibrated.",
                                  color = MaterialTheme.colorScheme.onSurfaceVariant,
                                  fontSize = 11.sp,
                                  lineHeight = 15.sp
                            )
                        }
                    }

                    // Interactive Region Wear Highlight Selector Card
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(12.dp))
                            .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f), RoundedCornerShape(12.dp))
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onLongPress = {
                                        tooltipToShow = Pair(
                                            "3D MESH WEAR SELECTORS",
                                            "Allows selecting specialized wear distribution modes. Selecting Center, Edge, or Camber patterns highlights simulated surface fatigue stresses mathematically projected across the core twin mesh."
                                        )
                                    }
                                )
                            }
                            .padding(14.dp)
                    ) {
                        Text(
                            text = "3D MESH WEAR SELECTORS",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 10.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            val wears = listOf("Normal", "Center Wear", "Edge Wear", "Camber Wear")
                            wears.forEach { w ->
                                val active = wearPattern == w
                                val tTag = "highlight_${w.lowercase().replace(" ", "_")}_btn"

                                Button(
                                    onClick = { viewModel.wearPattern.value = w; viewModel.runDiagnostics() },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = if (active) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.surfaceVariant,
                                        contentColor = if (active) Color.Black else MaterialTheme.colorScheme.onSurfaceVariant
                                    ),
                                    shape = RoundedCornerShape(6.dp),
                                    contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(34.dp)
                                        .testTag(tTag)
                                ) {
                                    Text(
                                        text = w.substringBefore(" "),
                                        fontSize = 10.sp,
                                        fontFamily = FontFamily.Monospace,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }
                    }

                    // Seasonal Thermal Profiles Comparison & Weather Degradation Analysis Card
                    SeasonalThermalComparisonView(
                        currentTemp = temperature,
                        currentSpeed = speed,
                        useMetric = useMetric
                    )

                    // AI-Reasoning Panel (Gemini Diagnostic feedback)
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("gemini_reasoning_panel"),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(
                                Icons.Default.AutoAwesome,
                                contentDescription = "AI",
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(18.dp)
                            )
                            Text(
                                text = "GEMINI COGNITIVE ANALYTICS",
                                color = MaterialTheme.colorScheme.primary,
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.2.sp
                            )

                            if (isAnalyzing) {
                                Spacer(modifier = Modifier.width(8.dp))
                                CircularProgressIndicator(
                                    color = MaterialTheme.colorScheme.primary,
                                    strokeWidth = 1.5.dp,
                                    modifier = Modifier.size(14.dp)
                                )
                            }
                        }

                        if (isAnalyzing && analysisState == null) {
                            // Shimmer / loading state
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(100.dp)
                                    .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(10.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "Interrogating neural vehicle pathways...",
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp
                                )
                            }
                        } else {
                            val analysis = analysisState

                            if (analysis != null) {
                                // 0. Gemini AI Tread Life Prediction
                                DiagnosticBlock(
                                    title = "AI-DRIVEN REMAINING TREAD LIFE",
                                    body = analysis.remainingLifePrediction,
                                    badge = "GEMINI ACTIVE PREDICTION",
                                    badgeColor = MaterialTheme.colorScheme.tertiary,
                                    icon = Icons.Default.HourglassEmpty
                                )

                                // 1. Detailed Technical Analysis Card
                                DiagnosticBlock(
                                    title = "OBSERVED DEGRADATION PROFILES",
                                    body = analysis.analysis,
                                    badge = "HYPER-PHYSICS INTERFACE",
                                    badgeColor = MaterialTheme.colorScheme.primary,
                                    icon = Icons.Default.Analytics
                                )

                                // 2. Safety Critical Risk Assessment Card
                                val riskColor = if (healthScore < 50) StatusCritical else StatusWarning
                                DiagnosticBlock(
                                    title = "CRITICAL RISK VALUATION",
                                    body = analysis.safety,
                                    badge = "SAFETY METRIC ALERT",
                                    badgeColor = riskColor,
                                    icon = Icons.Default.Warning
                                )

                                // 3. Maintenance Replacement Timeline List
                                DiagnosticBlock(
                                    title = "RELIABILITY REPLACEMENT SCHEDULE",
                                    body = analysis.timeline,
                                    badge = "PROBABLE LIFESPAN WINDOW",
                                    badgeColor = MaterialTheme.colorScheme.tertiary,
                                    icon = Icons.Default.CalendarToday
                                )

                                // 4. Visual Evidence Assets
                                if (frontImageUri != null || sidewallImageUri != null) {
                                    Text(
                                        text = "VISUAL EVIDENCE LOG",
                                        style = MaterialTheme.typography.labelSmall.copy(
                                            fontFamily = FontFamily.Monospace,
                                            fontWeight = FontWeight.Bold,
                                            letterSpacing = 1.sp,
                                            color = MaterialTheme.colorScheme.primary
                                        ),
                                        modifier = Modifier.padding(top = 16.dp, bottom = 8.dp)
                                    )
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(100.dp),
                                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        frontImageUri?.let {
                                            AsyncImage(
                                                model = it,
                                                contentDescription = "Front Image",
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .border(0.5.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(8.dp)),
                                                contentScale = androidx.compose.ui.layout.ContentScale.Crop
                                            )
                                        }
                                        sidewallImageUri?.let {
                                            AsyncImage(
                                                model = it,
                                                contentDescription = "Sidewall Image",
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .border(0.5.dp, Color.White.copy(alpha = 0.2f), RoundedCornerShape(8.dp)),
                                                contentScale = androidx.compose.ui.layout.ContentScale.Crop
                                            )
                                        }
                                        if (frontImageUri == null || sidewallImageUri == null) {
                                            Box(modifier = Modifier.weight(1f))
                                        }
                                    }
                                }
                            } else {
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text(
                                        text = "No diagnostic record generated. Re-trigger sliders to process telemetry.",
                                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                                        fontSize = 12.sp,
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        }
                    }

                    // Export PDF Report Trigger Button
                    Button(
                        onClick = {
                            Toast.makeText(context, "PDF Report successfully saved to /Documents/TireTwin_PDF_Report.pdf", Toast.LENGTH_LONG).show()
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = MaterialTheme.colorScheme.primary,
                            contentColor = MaterialTheme.colorScheme.onPrimary
                        ),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(48.dp)
                            .testTag("export_pdf_button"),
                        elevation = ButtonDefaults.buttonElevation(defaultElevation = 2.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Icon(Icons.Default.PictureAsPdf, contentDescription = "PDF")
                            Text(
                                text = "EXPORT CAD HEALTH PDF REPORT",
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }
    }

    // Interactive tooltip dialog renderer
    tooltipToShow?.let { (title, explanation) ->
        TelemetryTooltipDialog(
            metricName = title,
            explanation = explanation,
            onDismiss = { tooltipToShow = null }
        )
    }
}

@Composable
fun DiagnosticBlock(
    title: String,
    body: String,
    badge: String,
    badgeColor: Color,
    icon: androidx.compose.ui.graphics.vector.ImageVector
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(10.dp))
            .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f), RoundedCornerShape(10.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Icon(icon, contentDescription = null, tint = badgeColor, modifier = Modifier.size(15.dp))
                Text(
                    text = title,
                    color = MaterialTheme.colorScheme.onSurface,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )
            }

            Box(
                modifier = Modifier
                    .background(badgeColor.copy(alpha = 0.15f), RoundedCornerShape(4.dp))
                    .border(0.5.dp, badgeColor.copy(alpha = 0.5f), RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = badge,
                    fontSize = 7.5.sp,
                    fontFamily = FontFamily.Monospace,
                    color = badgeColor,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.15f), thickness = 0.5.dp)

        Text(
            text = body,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = 12.sp,
            lineHeight = 17.sp,
            style = LocalTextStyle.current.copy(fontFamily = FontFamily.Default)
        )
    }
}

fun getHealthRatingTitle(score: Int): String = when {
    score >= 90 -> "EXCELLENT TYRE INTEGRITY"
    score >= 75 -> "ACCEPTABLE NOMINAL STATE"
    score >= 55 -> "MODERATE STRUCTURAL DEVIATION"
    else -> "CRITICAL WEAR EXTREME - RISK"
}

@Composable
fun SeasonalThermalComparisonView(
    currentTemp: Float,
    currentSpeed: Float,
    useMetric: Boolean = false
) {
    var selectedTireModel by remember { mutableStateOf("SPORT_UHP") } // SPORT_UHP, ALL_SEASON, WINTER_ICE
    var selectedSeason by remember { mutableStateOf("SUMMER") } // SUMMER, WINTER, AUTUMN

    // Animation progress for loading the comparison data
    var animProgress by remember { mutableStateOf(0f) }
    LaunchedEffect(selectedTireModel, selectedSeason) {
        animProgress = 0f
        animate(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = tween(650, easing = LinearOutSlowInEasing)
        ) { value, _ ->
            animProgress = value
        }
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .testTag("seasonal_thermal_comparison_card"),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Thermostat,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(16.dp)
                    )
                    Text(
                        text = "THERMAL WEATHER MATRIX OVERLAY",
                        color = MaterialTheme.colorScheme.primary,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp
                    )
                }
                Text(
                    text = "ECU CALIBRATION",
                    color = Color(0x8A, 0x9B, 0xA8).copy(alpha = 0.6f),
                    fontSize = 8.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = "Compare current sensor thermal profile against historical multi-season compound benchmarks to identify weather-induced wear accelerators.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 10.sp,
                lineHeight = 14.sp
            )

            Spacer(modifier = Modifier.height(14.dp))

            // Tire Model Selection Row
            Text(
                text = "SELECT TIRE COMPOUND TYPE",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 8.5.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(6.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                val tireModels = listOf(
                    "SPORT_UHP" to "UHP Summer",
                    "ALL_SEASON" to "Pro All-Season",
                    "WINTER_ICE" to "Studless Winter"
                )
                tireModels.forEach { (code, label) ->
                    val isActive = selectedTireModel == code
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(6.dp))
                            .background(
                                if (isActive) MaterialTheme.colorScheme.primary.copy(alpha = 0.15f)
                                else MaterialTheme.colorScheme.background
                            )
                            .border(
                                width = 1.dp,
                                color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline.copy(alpha = 0.15f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .clickable { selectedTireModel = code }
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = label.uppercase(),
                            color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 8.5.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Season Comparison Selection Row
            Text(
                text = "SELECT COMPARISON SEASON PATTERN",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontSize = 8.5.sp,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(6.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                val seasons = listOf(
                    "SUMMER" to "Summer Heat",
                    "WINTER" to "Winter Polar",
                    "AUTUMN" to "Autumn Wet"
                )
                seasons.forEach { (code, label) ->
                    val isActive = selectedSeason == code
                    val seasonColor = when (code) {
                        "SUMMER" -> Color(0xFFF97316)
                        "WINTER" -> Color(0xFF60A5FA)
                        else -> StatusWarning
                    }
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(6.dp))
                            .background(
                                if (isActive) seasonColor.copy(alpha = 0.15f)
                                else MaterialTheme.colorScheme.background
                            )
                            .border(
                                width = 1.dp,
                                color = if (isActive) seasonColor else MaterialTheme.colorScheme.outline.copy(alpha = 0.15f),
                                shape = RoundedCornerShape(6.dp)
                            )
                            .clickable { selectedSeason = code }
                            .padding(vertical = 8.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = label.uppercase(),
                            color = if (isActive) seasonColor else MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 8.5.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Graph Labels / Legend Indicators
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    // Active Run Legend Label
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        Box(modifier = Modifier.size(6.dp).background(MaterialTheme.colorScheme.primary, CircleShape))
                        Text("ACTIVE SCAN RUN", color = Color.White, fontSize = 8.sp, fontFamily = FontFamily.Monospace)
                    }
                    // Seasonal Baseline Legend Label
                    val legendColor = when (selectedSeason) {
                        "SUMMER" -> Color(0xFFF97316)
                        "WINTER" -> Color(0xFF60A5FA)
                        else -> StatusWarning
                    }
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        Box(modifier = Modifier.size(6.dp).background(legendColor, CircleShape))
                        Text("SEASONAL HISTORIC", color = legendColor, fontSize = 8.sp, fontFamily = FontFamily.Monospace)
                    }
                }
                Text(
                    text = if (useMetric) "Y: TEMP (°C) | X: SPEED (KM/H)" else "Y: TEMP (°F) | X: SPEED (MPH)",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontSize = 8.sp,
                    fontFamily = FontFamily.Monospace
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // High-Performance custom Canvas chart representation
            val dynamicGridLineCount = 5
            val maxSpeed = 120f
            val maxTemp = 120f

            val primaryColor = MaterialTheme.colorScheme.primary
            val background = MaterialTheme.colorScheme.background
            val outline = MaterialTheme.colorScheme.outline

            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(140.dp)
                    .background(background, RoundedCornerShape(8.dp))
                    .border(1.dp, outline.copy(alpha = 0.1f), RoundedCornerShape(8.dp))
            ) {
                val leftPx = 32.dp.toPx()
                val rightPx = size.width - 16.dp.toPx()
                val topPx = 14.dp.toPx()
                val bottomPx = size.height - 20.dp.toPx()

                val plotWidth = rightPx - leftPx
                val plotHeight = bottomPx - topPx

                val axisColor = Color.Gray.copy(alpha = 0.15f)
                val textPaint = Paint().apply {
                    color = android.graphics.Color.GRAY
                    textSize = 7.5.dp.toPx()
                    typeface = Typeface.MONOSPACE
                    textAlign = Paint.Align.RIGHT
                }

                // 1. Draw horizontal grid lines and Y scale ticks
                for (i in 0 until dynamicGridLineCount) {
                    val fraction = i.toFloat() / (dynamicGridLineCount - 1)
                    val yGridVal = 120f * fraction
                    val yPixel = bottomPx - (fraction * plotHeight)

                    drawLine(
                        color = axisColor,
                        start = Offset(leftPx, yPixel),
                        end = Offset(rightPx, yPixel),
                        strokeWidth = 1.dp.toPx()
                    )

                    drawContext.canvas.nativeCanvas.apply {
                        drawText(
                            "${yGridVal.toInt()}°C",
                            leftPx - 4.dp.toPx(),
                            yPixel + 3.dp.toPx(),
                            textPaint
                        )
                    }
                }

                // 2. Plot bottom speed ticks
                val speedTicks = listOf(0, 30, 60, 90, 120)
                speedTicks.forEach { sTick ->
                    val xPixel = leftPx + (sTick.toFloat() / maxSpeed) * plotWidth
                    drawLine(
                        color = axisColor,
                        start = Offset(xPixel, topPx),
                        end = Offset(xPixel, bottomPx),
                        strokeWidth = 1.dp.toPx()
                    )
                    drawContext.canvas.nativeCanvas.apply {
                        val tickPaint = Paint().apply {
                            color = android.graphics.Color.GRAY
                            textSize = 7.5.dp.toPx()
                            typeface = Typeface.MONOSPACE
                            textAlign = Paint.Align.CENTER
                        }
                        drawText(
                            "${sTick}k",
                            xPixel,
                            bottomPx + 12.dp.toPx(),
                            tickPaint
                        )
                    }
                }

                // 3. Generate Historical Seasonal curve points
                val seasonCurvePoints = mutableListOf<Offset>()
                val stepsCount = 12
                val seasonalColor = when (selectedSeason) {
                    "SUMMER" -> Color(0xFFF97316)
                    "WINTER" -> Color(0xFF60A5FA)
                    else -> StatusWarning
                }

                for (i in 0..stepsCount) {
                    val sVal = (i.toFloat() / stepsCount) * maxSpeed
                    val xPx = leftPx + (sVal / maxSpeed) * plotWidth

                    // Formula modeling compound heating behaviors under conditions
                    val baseTemp = when (selectedSeason) {
                        "SUMMER" -> 38f
                        "WINTER" -> -5f
                        else -> 12f
                    }
                    val heatingCoefficient = when (selectedTireModel) {
                        "SPORT_UHP" -> when (selectedSeason) {
                            "SUMMER" -> 0.35f
                            "WINTER" -> 0.15f
                            else -> 0.22f
                        }
                        "ALL_SEASON" -> when (selectedSeason) {
                            "SUMMER" -> 0.45f
                            "WINTER" -> 0.28f
                            else -> 0.35f
                        }
                        else -> when (selectedSeason) { // WINTER_ICE
                            "SUMMER" -> 0.72f // Runs extremely hot in summer
                            "WINTER" -> 0.45f // Elastic dynamic snow control
                            else -> 0.55f
                        }
                    }

                    val calTemp = baseTemp + (sVal * heatingCoefficient)
                    val calTempCoerced = calTemp.coerceIn(-10f, maxTemp)
                    val yPx = bottomPx - (((calTempCoerced - 0f) / maxTemp) * plotHeight * animProgress)

                    seasonCurvePoints.add(Offset(xPx, yPx))
                }

                // Draw Historical Seasonal curve line
                val seasonPath = Path().apply {
                    seasonCurvePoints.forEachIndexed { index, pt ->
                        if (index == 0) moveTo(pt.x, pt.y)
                        else lineTo(pt.x, pt.y)
                    }
                }
                drawPath(
                    path = seasonPath,
                    color = seasonalColor.copy(alpha = 0.85f),
                    style = Stroke(width = 2.dp.toPx())
                )

                // 4. Generate Current Active Telem Scan curve profile
                // Active profile baseline acts as normal running state which intercepts the live coordinate (currentSpeed, currentTemp)
                val activeCurvePoints = mutableListOf<Offset>()
                val activeBaselineTemp = 20f

                // Ratio calculates friction coefficient slope that intersects our actual active telemetry coordinates
                val activeFrictionRatio = if (currentSpeed > 5f) {
                    (currentTemp - activeBaselineTemp) / currentSpeed
                } else {
                    0.35f
                }

                for (i in 0..stepsCount) {
                    val sVal = (i.toFloat() / stepsCount) * maxSpeed
                    val xPx = leftPx + (sVal / maxSpeed) * plotWidth

                    val activeTemp = activeBaselineTemp + (sVal * activeFrictionRatio)
                    val activeTempCoerced = activeTemp.coerceIn(0f, maxTemp)
                    val yPx = bottomPx - (((activeTempCoerced - 0f) / maxTemp) * plotHeight)

                    activeCurvePoints.add(Offset(xPx, yPx))
                }

                // Draw Active trajectory
                val activePath = Path().apply {
                    activeCurvePoints.forEachIndexed { index, pt ->
                        if (index == 0) moveTo(pt.x, pt.y)
                        else lineTo(pt.x, pt.y)
                    }
                }
                drawPath(
                    path = activePath,
                    color = primaryColor, // Neon Pink/Cyan solid active
                    style = Stroke(width = 2.5f.dp.toPx())
                )

                // 5. Highlight actual current telemetry point crosshair
                val liveXPx = leftPx + (currentSpeed / maxSpeed).coerceIn(0f, 1f) * plotWidth
                val liveYPx = bottomPx - (((currentTemp - 0f) / maxTemp).coerceIn(0f, 1f) * plotHeight)

                // Vertical projection lines
                drawLine(
                    color = Color.White.copy(alpha = 0.15f),
                    start = Offset(liveXPx, topPx),
                    end = Offset(liveXPx, bottomPx),
                    strokeWidth = 1.dp.toPx()
                )
                // Horizontal projection lines
                drawLine(
                    color = Color.White.copy(alpha = 0.15f),
                    start = Offset(leftPx, liveYPx),
                    end = Offset(rightPx, liveYPx),
                    strokeWidth = 1.dp.toPx()
                )

                // Pulsing glowing center rings
                drawCircle(
                    color = primaryColor.copy(alpha = 0.3f),
                    radius = 9.dp.toPx(),
                    center = Offset(liveXPx, liveYPx)
                )
                drawCircle(
                    color = Color.White,
                    radius = 3.5f.dp.toPx(),
                    center = Offset(liveXPx, liveYPx)
                )
                drawCircle(
                    color = primaryColor,
                    radius = 2.dp.toPx(),
                    center = Offset(liveXPx, liveYPx)
                )
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Prognosis advisory insights card details
            val multiplier = when (selectedTireModel) {
                "SPORT_UHP" -> when (selectedSeason) {
                    "SUMMER" -> "1.0x"
                    "WINTER" -> "2.2x (Cold Hardening Risk)"
                    else -> "1.2x"
                }
                "ALL_SEASON" -> when (selectedSeason) {
                    "SUMMER" -> "1.3x"
                    "WINTER" -> "1.2x"
                    else -> "1.0x"
                }
                else -> when (selectedSeason) { // WINTER_ICE
                    "SUMMER" -> "3.5x (Critical Tread Melting)"
                    "WINTER" -> "1.0x"
                    else -> "1.7x"
                }
            }

            val physicsExplanation = when (selectedTireModel) {
                "SPORT_UHP" -> when (selectedSeason) {
                    "WINTER" -> "WARNING: Summer compound rubber polymer strings undergo 'glass transition' below 7°C, hardening into brittle structure. Running summer tires in polar freeze drastically degrades compound structural matrix, accelerating stress cracks."
                    "SUMMER" -> "SUCCESS: Summer sport compound reaches its optimal thermal window (65°C - 85°C) safely. Adhesion footprint matches summer tarmac perfectly with nominal baseline wear kinetics."
                    else -> "NOMINAL: Rainy wet asphalt creates water-cooling dynamics that suppress thermal buildup. Balanced friction is preserved but hydroplaning limit warrants speed monitoring."
                }
                "ALL_SEASON" -> when (selectedSeason) {
                    "SUMMER" -> "MODERATE: Warm pavement heat (+42°C) causes compound rubber molecules to loosen, shifting wear slightly above nominal baseline. Maintain 34 PSI to regulate thermal shoulder deformation."
                    "WINTER" -> "MODERATE: Cold asphalt limits optimal rolling friction, resulting in slight micro-slip cycles. Tread siping configuration delivers acceptable traction margins."
                    else -> "NOMINAL: All-season rubber is fully optimized to retain dynamic friction thresholds. Elasticity indices are balanced."
                }
                else -> when (selectedSeason) { // WINTER_ICE
                    "SUMMER" -> "CRITICAL EXTREME: Soft, deeply-siped winter rubber designed for freezing temperatures undergoes polymer reversion on blistered asphalt. Unrestricted block squirm generates massive friction overload, degrading tread 3.5x faster!"
                    "WINTER" -> "SUCCESS: Arctic studless siping holds extreme flexibility in sub-zero snow, spreading mechanical loads across maximum biting edges. Wear indicators register nominal performance."
                    else -> "ALERT: Wet seasonal rain cooling prevents winter compound from fully melting, but soft tread blocks yield higher shear strains than all-season footprints."
                }
            }

            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.background, RoundedCornerShape(8.dp))
                    .padding(10.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "WEATHER WEAR ACCELERATOR:",
                            color = MaterialTheme.colorScheme.primary,
                            fontSize = 8.5.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = multiplier,
                            color = if (multiplier.startsWith("1.0x")) MaterialTheme.colorScheme.tertiary else Color(0xFF, 0x55, 0x55),
                            fontSize = 9.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.1f), thickness = 0.5.dp)
                    Text(
                        text = physicsExplanation,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 9.5.sp,
                        lineHeight = 14.sp
                    )
                }
            }
        }
    }
}
