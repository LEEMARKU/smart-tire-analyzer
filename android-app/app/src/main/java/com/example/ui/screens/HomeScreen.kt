package com.example.ui.screens

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
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.layout.LayoutCoordinates
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.graphics.graphicsLayer
import android.widget.Toast
import com.example.ui.theme.StatusCritical
import com.example.ui.theme.StatusInfo
import com.example.ui.theme.StatusSuccess
import com.example.ui.theme.StatusWarning
import com.example.ui.components.TireDigitalTwin3D
import com.example.ui.components.TelemetryControlPanel
import com.example.ui.components.DiagnosticPanel
import com.example.ui.components.TelemetryTooltipDialog
import com.example.ui.components.AssetUploadSection
import com.example.util.PdfGenerator
import com.example.util.SnapshotHelper
import com.example.viewmodel.TireTwinViewModel
import com.example.viewmodel.TirePosition
import com.example.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    viewModel: TireTwinViewModel,
    onNavigateToCamera: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToResults: () -> Unit
) {
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
    val tireStates by viewModel.tireStates.collectAsState()
    val useMetric by viewModel.useMetricUnits.collectAsState()
    val notifications by viewModel.notifications.collectAsState()

    val context = LocalContext.current
    val view = LocalView.current
    val snackbarHostState = remember { SnackbarHostState() }
    var tooltipToShow by remember { mutableStateOf<Pair<String, String>?>(null) }
    var showCalibrationWizard by remember { mutableStateOf(false) }

    LaunchedEffect(notifications) {
        if (notifications.isNotEmpty()) {
            snackbarHostState.showSnackbar(
                message = notifications.last(),
                duration = SnackbarDuration.Long,
                withDismissAction = true
            )
            viewModel.clearNotifications()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) },
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageIcon(),
                            contentDescription = "App Logo",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(28.dp)
                        )
                        Column {
                            Text(
                                text = "TEK-TWIN 3D",
                                style = LocalTextStyle.current.copy(
                                    fontFamily = FontFamily.Monospace,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 17.sp,
                                    letterSpacing = 2.sp,
                                    color = MaterialTheme.colorScheme.onBackground
                                )
                            )
                            Text(
                                text = "Automotive Telemetry Engine",
                                style = LocalTextStyle.current.copy(
                                    fontSize = 10.sp,
                                    letterSpacing = 0.5.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            )
                        }
                    }
                },
                actions = {
                    IconButton(
                        onClick = {
                            com.example.util.CsvExporter.exportScansToCsv(context, viewModel.historyScans.value)
                        },
                        modifier = Modifier
                            .testTag("export_csv_button")
                            .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                            .size(38.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Dataset,
                            contentDescription = "Export CSV Data",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
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
                        modifier = Modifier
                            .testTag("export_pdf_button")
                            .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                            .size(38.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.PictureAsPdf,
                            contentDescription = "Export PDF Report",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = { viewModel.toggleUnits() },
                        modifier = Modifier
                            .testTag("unit_toggle_button")
                            .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                            .size(38.dp)
                    ) {
                        Text(
                            text = if (useMetric) "M" else "I",
                            color = MaterialTheme.colorScheme.primary,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = { viewModel.toggleTheme() },
                        modifier = Modifier
                            .testTag("theme_toggle_button")
                            .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                            .size(38.dp)
                    ) {
                        Icon(
                            imageVector = if (isDarkTheme) Icons.Default.LightMode else Icons.Default.DarkMode,
                            contentDescription = "Toggle Theme",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.width(8.dp))
                    IconButton(
                        onClick = onNavigateToHistory,
                        modifier = Modifier
                            .testTag("history_nav_button")
                            .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape)
                            .size(38.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.History,
                            contentDescription = "History Log",
                            tint = MaterialTheme.colorScheme.primary
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.background,
                    titleContentColor = MaterialTheme.colorScheme.onBackground
                )
            )
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = onNavigateToCamera,
                containerColor = MaterialTheme.colorScheme.tertiary,
                contentColor = MaterialTheme.colorScheme.onTertiary,
                modifier = Modifier
                    .testTag("scan_floating_button")
                    .padding(8.dp)
                    .shadow(12.dp, CircleShape),
                shape = CircleShape
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.QrCodeScanner,
                        contentDescription = "Scanner",
                        modifier = Modifier.size(24.dp)
                    )
                    Text(
                        text = "SCAN PROFILE",
                        style = LocalTextStyle.current.copy(
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            fontFamily = FontFamily.Monospace,
                            letterSpacing = 1.sp
                        )
                    )
                }
            }
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Live Status Header Board
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
                    .testTag("health_telemetry_card"),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(14.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                     Column(
                        modifier = Modifier
                            .weight(1f)
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onLongPress = {
                                        tooltipToShow = Pair(
                                            "TREAD DEPTH",
                                            "Current depth of the tread grooves. Standard threshold is 5.2mm nominal; less than 1.6mm legal minimum triggers severe aquaplaning and low-friction replacement alert."
                                        )
                                    }
                                )
                            }
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .background(MaterialTheme.colorScheme.tertiary, CircleShape) // glowing status
                            )
                            Text(
                                text = "IoT SYNCED ON AIR",
                                color = MaterialTheme.colorScheme.tertiary,
                                fontSize = 10.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Model S Plaid (${activeTire.label})",
                            color = MaterialTheme.colorScheme.onSurface,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        val displayDepth = if (useMetric) "5.2mm" else "0.20in"
                        Text(
                            text = "Tread Depth: $displayDepth nominal",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 11.sp
                        )
                    }

                    // Score indicator badge
                    Column(
                        horizontalAlignment = Alignment.End,
                        modifier = Modifier
                            .padding(start = 8.dp)
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
                    ) {
                        Text(
                            text = "Overall Health",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 10.sp,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "$healthScore%",
                            color = evaluateHealthColor(healthScore),
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Black,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            // Multi-Tire Position Switcher & Parallel Sensor HUD
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 20.dp, end = 20.dp, top = 12.dp, bottom = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ACTIVE SENSOR TELEMETRY GATEWAY",
                    style = MaterialTheme.typography.labelSmall.copy(
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp,
                        color = MaterialTheme.colorScheme.primary
                    )
                )
                TextButton(
                    onClick = { showCalibrationWizard = true },
                    modifier = Modifier.testTag("calibrate_sensors_btn"),
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Wifi,
                        contentDescription = "Sync",
                        modifier = Modifier.size(14.dp),
                        tint = Color(0x50, 0xFA, 0x7B)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "SYNC SENSOR",
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 6.dp)
                    .testTag("multi_tire_sensor_hud"),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                TirePosition.values().forEach { pos ->
                    val state = tireStates[pos]
                    val isActive = pos == activeTire
                    val score = state?.let { viewModel.calculateHealthScoreFor(it.pressure, it.temperature, it.wearPattern) } ?: 100
                    val healthColor = evaluateHealthColor(score)

                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .clickable { viewModel.setActiveTire(pos) }
                            .testTag("tire_select_${pos.code.lowercase()}"),
                        shape = RoundedCornerShape(12.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = if (isActive) {
                                MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f)
                            } else {
                                MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.15f)
                            }
                        ),
                        border = BorderStroke(
                            width = if (isActive) 2.dp else 1.dp,
                            color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                        )
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(10.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = pos.code,
                                fontWeight = FontWeight.Black,
                                fontSize = 14.sp,
                                fontFamily = FontFamily.Monospace,
                                color = if (isActive) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = if (useMetric) {
                                    "${(state?.pressure?.times(0.0689f))?.toInt() ?: 2} BAR"
                                } else {
                                    "${state?.pressure?.toInt() ?: 32} PSI"
                                },
                                fontSize = 11.sp,
                                fontWeight = FontWeight.SemiBold,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            Spacer(modifier = Modifier.height(2.dp))
                            Text(
                                text = if (useMetric) {
                                    "${state?.temperature?.toInt() ?: 38}°C"
                                } else {
                                    "${(state?.temperature?.times(9 / 5f)?.plus(32))?.toInt() ?: 100}°F"
                                },
                                fontSize = 9.sp,
                                fontFamily = FontFamily.Monospace,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            // Status Dot & Health Rating
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(6.dp)
                                        .background(healthColor, CircleShape)
                                )
                                Text(
                                    text = "$score%",
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = FontFamily.Monospace,
                                    color = healthColor
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Interactive 3D Digital Twin Visualizer Pane
            var viewportCoordinates by remember { mutableStateOf<androidx.compose.ui.layout.LayoutCoordinates?>(null) }
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(300.dp)
                    .padding(horizontal = 16.dp)
                    .clip(RoundedCornerShape(16.dp))
                    .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
                    .onGloballyPositioned { viewportCoordinates = it }
            ) {
                // The rotating mathematical 3D rendering
                TireDigitalTwin3D(
                    speed = speed,
                    pressure = pressure,
                    temperature = temperature,
                    wearPattern = wearPattern,
                    isThermalMode = isThermalMode,
                    isExplodedView = isExplodedView,
                    isDarkTheme = isDarkTheme,
                    useMetric = useMetric
                )

                // Real-time battery level overlay of the IoT sensor module (Top-Left corner / TopStart)
                val iotBattery by viewModel.iotBatteryLevel.collectAsState()
                
                Box(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(12.dp)
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
                        .testTag("viewport_sensor_battery_widget")
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
                                .background(Color(0x32, 0xD7, 0x82).copy(alpha = alpha), CircleShape)
                        )
                        
                        Text(
                            text = "SENSOR BATT:",
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

                // Render floating visual overlay toggles inside canvas
                Column(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = { viewModel.isThermalMode.value = !isThermalMode },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isThermalMode) StatusWarning else Color(0x1F, 0x24, 0x2C).copy(alpha = 0.85f),
                            contentColor = Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier
                            .height(30.dp)
                            .testTag("thermal_heatmap_toggle"),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Thermostat,
                            contentDescription = "Thermal Mode",
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("THERMAL HEAT", fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                    }

                    Button(
                        onClick = { viewModel.isExplodedView.value = !isExplodedView },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (isExplodedView) MaterialTheme.colorScheme.primary else Color(0x1F, 0x24, 0x2C).copy(alpha = 0.85f),
                            contentColor = Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier
                            .height(30.dp)
                            .testTag("exploded_mesh_toggle"),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Layers,
                            contentDescription = "Exploded View",
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("EXPLODED VIEW", fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                    }

                    Button(
                        onClick = {
                            viewportCoordinates?.let { coords ->
                                val activeLabel = activeTire.code
                                val wearName = wearPattern.replace(" ", "_")
                                SnapshotHelper.captureAndSaveSnapshot(
                                    context = context,
                                    view = view,
                                    coordinates = coords,
                                    fileName = "TireTwin_${activeLabel}_${wearName}_${System.currentTimeMillis()}"
                                )
                            } ?: run {
                                Toast.makeText(context, "Viewport not ready for snapshot", Toast.LENGTH_SHORT).show()
                            }
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0x1F, 0x24, 0x2C).copy(alpha = 0.85f),
                            contentColor = Color.White
                        ),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 6.dp),
                        modifier = Modifier
                            .height(30.dp)
                            .testTag("capture_snapshot_button"),
                        shape = RoundedCornerShape(6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.CameraAlt,
                            contentDescription = "Capture Snapshot",
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("SNAPSHOT", fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                    }
                }

                // Show running wheel speedometer inside canvas
                Column(
                    modifier = Modifier
                        .align(Alignment.BottomStart)
                        .pointerInput(Unit) {
                            detectTapGestures(
                                onLongPress = {
                                    tooltipToShow = Pair(
                                        "ROTATIONAL FREQUENCY",
                                        "Real-time rotational frequency translated into vehicle velocity equivalents. High motor-driven speed subjects the core tread elements to extreme centrifugal deformation and shear temperature builds."
                                    )
                                }
                            )
                        }
                        .padding(14.dp)
                ) {
                    Text(
                        text = "ROTATIONAL FREQUENCY",
                        color = Color(0x8A, 0x9B, 0xA8).copy(alpha = 0.7f),
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = if (useMetric) "${speed.toInt()} Km/h" else "${(speed * 0.621f).toInt()} Mph",
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            // Interactive Telemetry Control Panel with Custom Presets & IoT Sim Feed
            TelemetryControlPanel(
                speed = speed,
                pressure = pressure,
                temperature = temperature,
                wearPattern = wearPattern,
                useMetric = useMetric,
                onTelemetryChange = { s, p, t ->
                    viewModel.updateTelemetry(s, p, t, wearPattern)
                },
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 14.dp)
            )

            // Wear Pattern Selector Mode Card
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 2.dp)
                    .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(12.dp))
                    .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.15f), RoundedCornerShape(12.dp))
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onLongPress = {
                                tooltipToShow = Pair(
                                    "OBSERVED WEAR PATTERN",
                                    "The geometric distribution of wear across the tire tread profile. Camber, Cupping, Center, or Edge wear indicate specific suspension misalignments or chronic inflation mistakes."
                                )
                            }
                        )
                    }
                    .padding(16.dp)
            ) {
                Text(
                    text = "OBSERVED WEAR PATTERN COUPLING",
                    color = MaterialTheme.colorScheme.primary,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 1.sp
                )
                Spacer(modifier = Modifier.height(10.dp))

                // Scrollable row of patterns
                val patterns = listOf("Normal", "Center Wear", "Edge Wear", "Camber Wear", "Cupping Wear")
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    patterns.forEach { pattern ->
                        val isSelected = wearPattern == pattern
                        val tagValue = "wear_${pattern.lowercase().replace(" ", "_")}_chip"

                        FilterChip(
                            selected = isSelected,
                            onClick = { viewModel.updateTelemetry(speed, pressure, temperature, pattern) },
                            label = { Text(pattern, fontSize = 11.sp, fontFamily = FontFamily.Monospace) },
                            colors = FilterChipDefaults.filterChipColors(
                                containerColor = MaterialTheme.colorScheme.background,
                                labelColor = MaterialTheme.colorScheme.onSurfaceVariant,
                                selectedContainerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.15f),
                                selectedLabelColor = MaterialTheme.colorScheme.primary
                            ),
                            border = FilterChipDefaults.filterChipBorder(
                                enabled = true,
                                selected = isSelected,
                                borderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f),
                                selectedBorderColor = MaterialTheme.colorScheme.primary,
                                borderWidth = 1.dp,
                                selectedBorderWidth = 1.2.dp
                            ),
                            modifier = Modifier.testTag(tagValue)
                        )
                    }
                }
            }

            // Real-time AI Degradation Prognosis card with frosted glass theme
            AssetUploadSection(
                viewModel = viewModel,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 6.dp)
            )

            DiagnosticPanel(
                speed = speed,
                pressure = pressure,
                temperature = temperature,
                wearPattern = wearPattern,
                analysisState = analysisState,
                isAnalyzing = isAnalyzing,
                useMetric = useMetric,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)
            )

            // Quick Diagnostic Action Button
            Button(
                onClick = onNavigateToResults,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
                    .height(48.dp)
                    .testTag("analyze_results_button"),
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0x21, 0x26, 0x2D),
                    contentColor = Color.White
                ),
                border = BorderStroke(1.dp, Color(0x30, 0x36, 0x3D))
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(Icons.Default.Analytics, contentDescription = "Analyze")
                    Text(
                        text = "VIEW DIAGNOSTICS & AI RUN",
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        letterSpacing = 1.sp
                    )
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
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

    if (showCalibrationWizard) {
        CalibrationWizardDialog(
            viewModel = viewModel,
            onDismiss = { showCalibrationWizard = false }
        )
    }
}

// Helpers
fun getPressureLabel(psi: Float): String = when {
    psi < 27.5f -> "(Under-Inflated)"
    psi > 36.5f -> "(Over-Inflated)"
    else -> "(Standard)"
}

fun getPressureColor(psi: Float): Color = when {
    psi < 27.5f -> StatusWarning
    psi > 36.5f -> StatusWarning
    else -> StatusSuccess
}

fun getTemperatureColor(temp: Float): Color = when {
    temp > 80f -> StatusCritical
    temp > 50f -> StatusWarning
    else -> StatusInfo
}

fun evaluateHealthColor(score: Int): Color = when {
    score >= 80 -> StatusSuccess
    score >= 50 -> StatusWarning
    else -> StatusCritical
}

// Icon helper function that avoids using R.drawable.xxx that may not compile
@Composable
fun imageIcon() = Icons.Default.Adjust

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CalibrationWizardDialog(
    viewModel: TireTwinViewModel,
    onDismiss: () -> Unit
) {
    var currentStep by remember { mutableStateOf(1) }
    var selectedPosition by remember { mutableStateOf(TirePosition.FRONT_LEFT) }
    
    // Scan animation states
    var isScanning by remember { mutableStateOf(false) }
    var scanProgress by remember { mutableStateOf(0f) }
    var sensorFound by remember { mutableStateOf(false) }
    
    // Calibration animation states
    var calibrating by remember { mutableStateOf(false) }
    var calibrationProgress by remember { mutableStateOf(0f) }
    var calibrationStatusText by remember { mutableStateOf("Ready to zero sensor...") }
    var initialPressure by remember { mutableStateOf(32f) }

    val context = LocalContext.current

    // Simulated scan execution
    LaunchedEffect(isScanning) {
        if (isScanning) {
            scanProgress = 0f
            sensorFound = false
            val duration = 3000L
            val stepTime = 50L
            val steps = duration / stepTime
            for (i in 1..steps.toInt()) {
                kotlinx.coroutines.delay(stepTime)
                scanProgress = i.toFloat() / steps
            }
            sensorFound = true
            isScanning = false
        }
    }

    // Simulated calibration execution
    LaunchedEffect(calibrating) {
        if (calibrating) {
            calibrationProgress = 0f
            val steps = 30
            val statusTexts = listOf(
                "Reading ambient temperature...",
                "Zeroing pressure transducers...",
                "Storing baseline calibration curve...",
                "Synchronizing with digital twin model..."
            )
            for (i in 1..steps) {
                kotlinx.coroutines.delay(100L)
                calibrationProgress = i.toFloat() / steps
                val textIdx = ((i.toFloat() / steps) * statusTexts.size).toInt().coerceAtMost(statusTexts.size - 1)
                calibrationStatusText = statusTexts[textIdx]
            }
            calibrating = false
            currentStep = 4 // Go to Success
        }
    }

    AlertDialog(
        onDismissRequest = { if (!calibrating && !isScanning) onDismiss() },
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .testTag("calibration_wizard_dialog"),
        properties = androidx.compose.ui.window.DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = Color.White.copy(alpha = 0.08f)),
            border = BorderStroke(1.dp, Color(0x30, 0x36, 0x3D).copy(alpha = 0.8f)),
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0x0D, 0x11, 0x17).copy(alpha = 0.95f))
                    .padding(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Header with title & close
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "SENSOR CALIBRATION WIZARD",
                        color = MaterialTheme.colorScheme.primary,
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.5.sp
                    )
                    if (!calibrating && !isScanning) {
                        IconButton(onClick = onDismiss, modifier = Modifier.size(24.dp)) {
                            Icon(Icons.Default.Close, contentDescription = "Close", tint = Color.Gray, modifier = Modifier.size(16.dp))
                        }
                    }
                }

                // Wizard Steps indicator bar
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    for (step in 1..4) {
                        val isActive = step == currentStep
                        val isDone = step < currentStep
                        val barColor = when {
                            isActive -> MaterialTheme.colorScheme.primary
                            isDone -> MaterialTheme.colorScheme.tertiary
                            else -> Color(0x21, 0x26, 0x2D)
                        }
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .height(4.dp)
                                .clip(RoundedCornerShape(2.dp))
                                .background(barColor)
                        )
                    }
                }

                // Dynamic Body Content based on Step
                when (currentStep) {
                    1 -> { // Step 1: Position Selection
                        Text(
                            text = "STEP 1: SELECT TIRE POSITION",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Select the wheel position you are mounting the new IoT hardware sensor onto:",
                            color = Color(0x8A, 0x9B, 0xA8),
                            fontSize = 10.sp,
                            textAlign = TextAlign.Center,
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 14.sp
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        // Visual Chassis Layout Selector
                        Box(
                            modifier = Modifier
                                .width(160.dp)
                                .height(220.dp)
                                .border(1.dp, Color(0x21, 0x26, 0x2D), RoundedCornerShape(8.dp))
                                .padding(12.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            // Simple chassis line representation
                            Canvas(modifier = Modifier.fillMaxSize()) {
                                val w = size.width
                                val h = size.height
                                // Draw main vertical driveshaft spine
                                drawLine(Color(0x21, 0x26, 0x2D), Offset(w/2, h*0.15f), Offset(w/2, h*0.85f), strokeWidth = 6f)
                                // Front axle
                                drawLine(Color(0x21, 0x26, 0x2D), Offset(w*0.2f, h*0.25f), Offset(w*0.8f, h*0.25f), strokeWidth = 4f)
                                // Rear axle
                                drawLine(Color(0x21, 0x26, 0x2D), Offset(w*0.2f, h*0.75f), Offset(w*0.8f, h*0.75f), strokeWidth = 4f)
                            }

                            // Interactive tire positions overlay buttons
                            Column(
                                modifier = Modifier.fillMaxSize(),
                                verticalArrangement = Arrangement.SpaceBetween
                            ) {
                                // Front Row
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    ChassisTireButton(TirePosition.FRONT_LEFT, selectedPosition) { selectedPosition = it }
                                    ChassisTireButton(TirePosition.FRONT_RIGHT, selectedPosition) { selectedPosition = it }
                                }
                                // Rear Row
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    ChassisTireButton(TirePosition.REAR_LEFT, selectedPosition) { selectedPosition = it }
                                    ChassisTireButton(TirePosition.REAR_RIGHT, selectedPosition) { selectedPosition = it }
                                }
                            }
                        }

                        Button(
                            onClick = { currentStep = 2 },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text("PROCEED TO SCAN", fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                        }
                    }
                    2 -> { // Step 2: Signal Sync/Radar
                        Text(
                            text = "STEP 2: DETECT WIRELESS SIGNAL",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )

                        if (!isScanning && !sensorFound) {
                            Text(
                                text = "Hold the sensor near your mobile device. Click scan to search for the BLE calibration beacon broadcast.",
                                color = Color(0x8A, 0x9B, 0xA8),
                                fontSize = 10.sp,
                                textAlign = TextAlign.Center,
                                fontFamily = FontFamily.Monospace,
                                lineHeight = 14.sp
                            )
                            
                            Spacer(modifier = Modifier.height(16.dp))

                            Button(
                                onClick = { isScanning = true },
                                modifier = Modifier.fillMaxWidth(),
                                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Text("START PAIRING SCAN", fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                            }
                        } else if (isScanning) {
                            // Pulsing Radar scan visuals
                            Box(
                                modifier = Modifier
                                    .size(100.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                val infiniteTransition = rememberInfiniteTransition(label = "radar")
                                val scale by infiniteTransition.animateFloat(
                                    initialValue = 0.2f,
                                    targetValue = 1.0f,
                                    animationSpec = infiniteRepeatable(
                                        animation = tween(1500, easing = LinearEasing),
                                        repeatMode = RepeatMode.Restart
                                    ),
                                    label = "scale"
                                )
                                val alpha by infiniteTransition.animateFloat(
                                    initialValue = 0.8f,
                                    targetValue = 0.0f,
                                    animationSpec = infiniteRepeatable(
                                        animation = tween(1500, easing = LinearEasing),
                                        repeatMode = RepeatMode.Restart
                                    ),
                                    label = "alpha"
                                )
                                Box(
                                    modifier = Modifier
                                        .size(100.dp)
                                        .align(Alignment.Center)
                                        .graphicsLayer(scaleX = scale, scaleY = scale, alpha = alpha)
                                        .background(MaterialTheme.colorScheme.primary, CircleShape)
                                )
                                Icon(
                                    Icons.Default.Wifi,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(36.dp)
                                )
                            }
                            
                            Spacer(modifier = Modifier.height(10.dp))
                            
                            Text(
                                text = "SCANNING FOR BEACON... ${(scanProgress * 100).toInt()}%",
                                color = MaterialTheme.colorScheme.primary,
                                fontSize = 10.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold
                            )
                            LinearProgressIndicator(
                                progress = scanProgress,
                                modifier = Modifier.fillMaxWidth().height(4.dp),
                                color = MaterialTheme.colorScheme.primary,
                                trackColor = Color(0x21, 0x26, 0x2D)
                            )
                        } else if (sensorFound) {
                            // Discovery UI
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(Color(0x28, 0xB4, 0x82).copy(alpha = 0.12f))
                                    .border(1.dp, Color(0x28, 0xB4, 0x82).copy(alpha = 0.4f), RoundedCornerShape(8.dp))
                                    .padding(14.dp)
                            ) {
                                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                        Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Color(0x28, 0xB4, 0x82), modifier = Modifier.size(16.dp))
                                        Text("SENSOR HARDWARE DETECTED!", color = Color(0x28, 0xB4, 0x82), fontSize = 10.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                                    }
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text("DEVICE: IoT-TireTwin-7D:9A:E2:BC", color = Color.White, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                                    Text("PROTOCOL: Bluetooth BLE v5.3", color = Color(0x8A, 0x9B, 0xA8), fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                                    Text("SIGNAL STRENGTH: -54 dBm (Excellent)", color = MaterialTheme.colorScheme.primary, fontSize = 9.sp, fontFamily = FontFamily.Monospace)
                                }
                            }

                            Spacer(modifier = Modifier.height(8.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                OutlinedButton(
                                    onClick = { sensorFound = false },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(6.dp),
                                    border = BorderStroke(1.dp, Color(0x21, 0x26, 0x2D))
                                ) {
                                    Text("RE-SCAN", fontSize = 10.sp, fontFamily = FontFamily.Monospace, color = Color.White)
                                }
                                Button(
                                    onClick = { currentStep = 3 },
                                    modifier = Modifier.weight(1f),
                                    colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                                    shape = RoundedCornerShape(6.dp)
                                ) {
                                    Text("CALIBRATE PSI", fontSize = 10.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                                }
                            }
                        }
                    }
                    3 -> { // Step 3: Initial PSI Calibration
                        Text(
                            text = "STEP 3: BENCHMARK BASELINE PRESSURE",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )

                        if (!calibrating) {
                            Text(
                                text = "Verify current physical tire pressure with a baseline gauge, and input it below to calibrate sensor reference offsets:",
                                color = Color(0x8A, 0x9B, 0xA8),
                                fontSize = 10.sp,
                                textAlign = TextAlign.Center,
                                fontFamily = FontFamily.Monospace,
                                lineHeight = 14.sp
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0x21, 0x26, 0x2D).copy(alpha = 0.5f))
                            ) {
                                Column(modifier = Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text("TARGET BASELINE PRESSURE", color = Color(0x8A, 0x9B, 0xA8), fontSize = 8.sp, fontFamily = FontFamily.Monospace)
                                    Text("${"%.1f".format(initialPressure)} PSI", color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                                    Slider(
                                        value = initialPressure,
                                        onValueChange = { initialPressure = it },
                                        valueRange = 20f..40f,
                                        steps = 40,
                                        colors = SliderDefaults.colors(
                                            thumbColor = MaterialTheme.colorScheme.primary,
                                            activeTrackColor = MaterialTheme.colorScheme.primary,
                                            inactiveTrackColor = Color(0x0D, 0x11, 0x17)
                                        )
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(8.dp))

                            Button(
                                onClick = { calibrating = true },
                                modifier = Modifier.fillMaxWidth(),
                                colors = ButtonDefaults.buttonColors(containerColor = Color(0x28, 0xB4, 0x82)),
                                shape = RoundedCornerShape(6.dp)
                            ) {
                                Text("CALIBRATE BASELINE", fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                            }
                        } else {
                            // Calibrating Progress Screen
                            CircularProgressIndicator(
                                progress = calibrationProgress,
                                modifier = Modifier.size(50.dp),
                                color = Color(0x28, 0xB4, 0x82),
                                strokeWidth = 4.dp
                            )
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(
                                text = calibrationStatusText,
                                color = Color.White,
                                fontSize = 10.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Medium,
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = "CALIBRATION CYCLE: ${(calibrationProgress * 100).toInt()}%",
                                color = Color(0x28, 0xB4, 0x82),
                                fontSize = 9.sp,
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    4 -> { // Step 4: Success Confirmation
                        Box(
                            modifier = Modifier
                                .size(70.dp)
                                .background(Color(0x28, 0xB4, 0x82).copy(alpha = 0.15f), CircleShape)
                                .border(1.dp, Color(0x28, 0xB4, 0x82), CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                Icons.Default.Check,
                                contentDescription = "Check",
                                tint = Color(0x28, 0xB4, 0x82),
                                modifier = Modifier.size(36.dp)
                            )
                        }

                        Text(
                            text = "CALIBRATION COMPLETE!",
                            color = Color(0x28, 0xB4, 0x82),
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "New IoT pressure sensor paired successfully with the ${selectedPosition.label} digital twin model, zeroed at $initialPressure PSI.",
                            color = Color(0x8A, 0x9B, 0xA8),
                            fontSize = 10.sp,
                            textAlign = TextAlign.Center,
                            fontFamily = FontFamily.Monospace,
                            lineHeight = 14.sp
                        )

                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = Color(0x21, 0x26, 0x2D).copy(alpha = 0.5f)),
                            border = BorderStroke(0.5.dp, Color.White.copy(alpha = 0.1f))
                        ) {
                            Column(
                                modifier = Modifier.padding(12.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                SensorResultRow(label = "Tire Position:", value = selectedPosition.label)
                                SensorResultRow(label = "Sensor MAC ID:", value = "IoT-TireTwin-7D:9A:E2:BC")
                                SensorResultRow(label = "Calibrated Base:", value = "$initialPressure PSI")
                                SensorResultRow(label = "Carcass Battery:", value = "100.0% (Synced)")
                                SensorResultRow(label = "Telemetry Status:", value = "ONLINE (Continuous)")
                            }
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        Button(
                            onClick = {
                                viewModel.calibrateSensor(selectedPosition, initialPressure)
                                onDismiss()
                            },
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0x28, 0xB4, 0x82)),
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text("COMPLETE WIZARD", fontSize = 11.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChassisTireButton(
    position: TirePosition,
    selected: TirePosition,
    onSelect: (TirePosition) -> Unit
) {
    val isSelected = position == selected
    val label = position.code
    Box(
        modifier = Modifier
            .size(width = 44.dp, height = 34.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(
                if (isSelected) MaterialTheme.colorScheme.primary
                else Color(0x21, 0x26, 0x2D)
            )
            .border(
                1.dp,
                if (isSelected) Color.White else Color.Transparent,
                RoundedCornerShape(4.dp)
            )
            .clickable { onSelect(position) }
            .testTag("chassis_select_${label.lowercase()}"),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            color = if (isSelected) Color.Black else Color.White,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun SensorResultRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = label, color = Color(0x8A, 0x9B, 0xA8), fontSize = 9.sp, fontFamily = FontFamily.Monospace)
        Text(text = value, color = Color.White, fontSize = 9.sp, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
    }
}
