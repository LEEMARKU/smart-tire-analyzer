package com.example.viewmodel

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.room.Room
import com.example.data.AppDatabase
import com.example.data.GeminiAnalysis
import com.example.data.GeminiService
import com.example.data.TireScan
import com.example.data.TireScanRepository
import com.example.util.TireInferenceEngine
import com.example.util.TireInferenceResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

enum class TirePosition(val code: String, val label: String) {
    FRONT_LEFT("FL", "Front Left"),
    FRONT_RIGHT("FR", "Front Right"),
    REAR_LEFT("RL", "Rear Left"),
    REAR_RIGHT("RR", "Rear Right")
}

data class TireSensorState(
    val position: TirePosition,
    val pressure: Float,
    val temperature: Float,
    val wearPattern: String,
    val iotBatteryLevel: Float,
    val analysis: GeminiAnalysis? = null,
    val isAnalyzing: Boolean = false
)

class TireTwinViewModel(application: Application) : AndroidViewModel(application) {

    // Room Database and Repository setup
    private val database by lazy {
        Room.databaseBuilder(
            application,
            AppDatabase::class.java,
            "tire_twin_db"
        ).fallbackToDestructiveMigration().build()
    }

    val repository by lazy {
        TireScanRepository(database.tireScanDao())
    }

    // List of historical scans populated from Room
    val historyScans: StateFlow<List<TireScan>> = repository.allScans
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    // Current active tire selection flow
    val activeTire = MutableStateFlow(TirePosition.FRONT_LEFT)

    // Track state of all 4 tires simultaneously
    val tireStates = MutableStateFlow<Map<TirePosition, TireSensorState>>(
        mapOf(
            TirePosition.FRONT_LEFT to TireSensorState(TirePosition.FRONT_LEFT, 32f, 38f, "Normal", 98.2f),
            TirePosition.FRONT_RIGHT to TireSensorState(TirePosition.FRONT_RIGHT, 34f, 42f, "Center Wear", 94.8f),
            TirePosition.REAR_LEFT to TireSensorState(TirePosition.REAR_LEFT, 25f, 58f, "Edge Wear", 88.5f),
            TirePosition.REAR_RIGHT to TireSensorState(TirePosition.REAR_RIGHT, 31f, 35f, "Normal", 92.1f)
        )
    )

    // Current interactive active sliders state (mirrors currently active tire)
    val speed = MutableStateFlow(45f)       // km/h (shared across vehicle wheels)
    val pressure = MutableStateFlow(32f)    // PSI (Standard 32)
    val temperature = MutableStateFlow(38f) // °C
    val wearPattern = MutableStateFlow("Normal") // Normal, Center Wear, Edge Wear, etc.

    val isThermalMode = MutableStateFlow(false)
    val isExplodedView = MutableStateFlow(false)
    
    // Global theme mode state flow
    val isDarkTheme = MutableStateFlow(true)
    
    // Global Unit Configuration (Metric: km/h, Bar, °C | Imperial: mph, PSI, °F)
    val useMetricUnits = MutableStateFlow(false) 
    
    // Authentication state
    val isLoggedIn = MutableStateFlow(false)
    val userEmail = MutableStateFlow("")

    private val _frontImageUri = MutableStateFlow<Uri?>(null)
    val frontImageUri: StateFlow<Uri?> = _frontImageUri.asStateFlow()

    private val _sidewallImageUri = MutableStateFlow<Uri?>(null)
    val sidewallImageUri: StateFlow<Uri?> = _sidewallImageUri.asStateFlow()

    fun setFrontImageUri(uri: Uri?) {
        _frontImageUri.value = uri
    }

    fun setSidewallImageUri(uri: Uri?) {
        _sidewallImageUri.value = uri
    }

    // Notification states
    private val _notifications = MutableStateFlow<List<String>>(emptyList())
    val notifications: StateFlow<List<String>> = _notifications.asStateFlow()

    fun toggleUnits() {
        useMetricUnits.value = !useMetricUnits.value
    }

    fun login(email: String) {
        userEmail.value = email
        isLoggedIn.value = true
    }

    fun logout() {
        isLoggedIn.value = false
    }

    // IoT Sensor Module real-time battery level state (starts at a realistic value)
    val iotBatteryLevel = MutableStateFlow(98.2f)
    
    fun toggleTheme() {
        isDarkTheme.value = !isDarkTheme.value
    }

    // AI Analysis states
    private val _analysisState = MutableStateFlow<GeminiAnalysis?>(null)
    val analysisState: StateFlow<GeminiAnalysis?> = _analysisState.asStateFlow()

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    // ONNX Runtime offline inference
    val tireInferenceEngine = TireInferenceEngine(application)
    private val _tireInferenceResult = MutableStateFlow<TireInferenceResult?>(null)
    val tireInferenceResult: StateFlow<TireInferenceResult?> = _tireInferenceResult.asStateFlow()

    fun runTireInference(bitmap: Bitmap) {
        viewModelScope.launch(Dispatchers.IO) {
            tireInferenceEngine.loadModel()
            val result = tireInferenceEngine.infer(bitmap)
            _tireInferenceResult.value = result
            _analysisState.value = GeminiAnalysis(
                analysis = "AI condition: ${result.condition} (${"%.0f".format(result.conditionConfidence * 100)}% confidence), health index: ${"%.0f".format(result.health * 100)}%, remaining life: ${"%.0f".format(result.remainingLife * 100)}%",
                safety = when (result.condition) {
                    "safe" -> "Low risk — tire is in good condition."
                    "moderate" -> "Medium risk — monitor tread wear and schedule inspection."
                    else -> "Critical — immediate replacement recommended."
                },
                timeline = when (result.condition) {
                    "safe" -> "Next inspection in 6 months."
                    "moderate" -> "Schedule inspection within 30 days."
                    else -> "Replace tire immediately."
                },
                remainingLifePrediction = "AI predicts ${"%.0f".format(result.remainingLife * 100)}% tread life remaining"
            )
            _isAnalyzing.value = false
            syncActiveTireToMap()
        }
    }

    // Calculated health rating based on metrics:
    val healthScore: StateFlow<Int> = combine(pressure, temperature, wearPattern) { pres, temp, wear ->
        calculateHealthScoreFor(pres, temp, wear)
    }.stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 100)

    fun calculateHealthScoreFor(pres: Float, temp: Float, wear: String): Int {
        var score = 100

        // Pressure impact
        val pDiff = kotlin.math.abs(pres - 32f)
        score -= (pDiff * 1.5f).toInt()

        // Temperature impact (> 65 generates high stress)
        if (temp > 65f) {
            score -= ((temp - 65f) * 0.8f).toInt()
        }

        // Wear pattern degradation index
        score -= when (wear) {
            "Center Wear" -> 22
            "Edge Wear" -> 25
            "Camber Wear" -> 30
            "Cupping Wear" -> 40
            else -> 0
        }

        return score.coerceIn(0, 100)
    }

    init {
        // Run initial diagnostics for Front Left
        runDiagnostics()

        // Simulate real-time IoT battery level depletion for all sensors simultaneously
        viewModelScope.launch {
            while (true) {
                kotlinx.coroutines.delay(2000) // Update every 2 seconds
                val currentBattery = iotBatteryLevel.value
                
                // Base speed and heat telemetry affects consumption slightly
                val baseConsumption = 0.003f
                val speedLoad = (speed.value / 150f) * 0.015f
                val tempLoad = (temperature.value / 100f) * 0.008f
                
                val delta = baseConsumption + speedLoad + tempLoad
                val nextBattery = currentBattery - delta
                val updatedActiveBattery = if (nextBattery < 5f) 100f else nextBattery
                
                // Add minor random fluctuation to emulate real-world live signal telemetry
                val noise = (Math.random().toFloat() - 0.5f) * 0.05f
                val finalActiveBattery = (updatedActiveBattery + noise).coerceIn(1f, 100f)
                iotBatteryLevel.value = finalActiveBattery
                
                // Update map for all 4 positions simultaneously
                val currentStates = tireStates.value.toMutableMap()
                currentStates.forEach { (pos, state) ->
                    val otherBattery = if (pos == activeTire.value) {
                         finalActiveBattery
                    } else {
                        val otherDelta = (baseConsumption + (speed.value / 150f) * 0.012f + (state.temperature / 100f) * 0.007f)
                        var nextOther = state.iotBatteryLevel - otherDelta
                        if (nextOther < 5f) nextOther = 100f
                        val otherNoise = (Math.random().toFloat() - 0.5f) * 0.03f
                        (nextOther + otherNoise).coerceIn(1f, 100f)
                    }
                    currentStates[pos] = state.copy(iotBatteryLevel = otherBattery)
                }
                tireStates.value = currentStates
            }
        }
    }

    fun syncActiveTireToMap() {
        val currentActive = activeTire.value
        val updatedStates = tireStates.value.toMutableMap()
        updatedStates[currentActive] = TireSensorState(
            position = currentActive,
            pressure = pressure.value,
            temperature = temperature.value,
            wearPattern = wearPattern.value,
            iotBatteryLevel = iotBatteryLevel.value,
            analysis = _analysisState.value,
            isAnalyzing = _isAnalyzing.value
        )
        tireStates.value = updatedStates
    }

    fun setActiveTire(position: TirePosition) {
        // Save current active tire state first
        syncActiveTireToMap()

        // Switch selection
        activeTire.value = position

        // Load targeted tire's state
        val targetState = tireStates.value[position] ?: TireSensorState(position, 32f, 38f, "Normal", 95f)
        pressure.value = targetState.pressure
        temperature.value = targetState.temperature
        wearPattern.value = targetState.wearPattern
        iotBatteryLevel.value = targetState.iotBatteryLevel
        _analysisState.value = targetState.analysis
        _isAnalyzing.value = targetState.isAnalyzing

        // Run diagnostic analysis if it doesn't already have one compiled
        if (targetState.analysis == null) {
            runDiagnostics()
        }
    }

    fun calibrateSensor(position: TirePosition, initialPressure: Float) {
        val currentStates = tireStates.value.toMutableMap()
        val oldState = currentStates[position] ?: TireSensorState(position, 32f, 38f, "Normal", 98f)
        
        // Set battery life back to 100% and sync initial pressure
        val newState = oldState.copy(
            pressure = initialPressure,
            temperature = 30f, // nominal baseline starting temperature
            iotBatteryLevel = 100f,
            wearPattern = "Normal"
        )
        currentStates[position] = newState
        tireStates.value = currentStates
        
        // If this is currently active, load into the interactive flow
        if (position == activeTire.value) {
            pressure.value = initialPressure
            temperature.value = 30f
            wearPattern.value = "Normal"
            iotBatteryLevel.value = 100f
            runDiagnostics()
        }
    }

    fun updateTelemetry(spd: Float, pres: Float, temp: Float, wear: String) {
        speed.value = spd
        pressure.value = pres
        temperature.value = temp
        wearPattern.value = wear
        
        // Sync to map
        syncActiveTireToMap()
        
        // Re-trigger analysis when telemetry changes
        runDiagnostics()
    }

    fun runDiagnostics() {
        viewModelScope.launch {
            _isAnalyzing.value = true
            syncActiveTireToMap()
            val result = GeminiService.getTireAnalysis(
                speed = speed.value,
                pressure = pressure.value,
                temperature = temperature.value,
                wearPattern = wearPattern.value
            )
            _analysisState.value = result
            _isAnalyzing.value = false
            syncActiveTireToMap()
            
            // AI-reasoning proactive maintenance check
            checkMaintenanceAlerts(result)
        }
    }

    private fun checkMaintenanceAlerts(analysis: GeminiAnalysis?) {
        // AI-reasoning based alerts for degradation patterns
        analysis?.let {
            if (it.analysis.contains("accelerated", ignoreCase = true) ||
                it.analysis.contains("degradation", ignoreCase = true) ||
                it.safety.contains("Critical", ignoreCase = true)) {
                
                val alert = "AI ALERT: Accelerated tread degradation detected on ${activeTire.value.label}. Immediate maintenance required."
                if (!_notifications.value.contains(alert)) {
                    _notifications.value = _notifications.value + alert
                }
            }
            
            // AI-driven maintenance timing suggestion
            if (it.remainingLifePrediction.contains("immediate", ignoreCase = true) || 
                it.timeline.contains("ASAP", ignoreCase = true)) {
                val suggestion = "MAINTENANCE SUGGESTION: ${it.timeline}"
                if (!_notifications.value.contains(suggestion)) {
                    _notifications.value = _notifications.value + suggestion
                }
            }
        }
        
        // Threshold-based proactive alerts
        if (pressure.value < 22f) {
            val alert = "CRITICAL: Severe under-inflation detected on ${activeTire.value.label}. Blowout risk high!"
            if (!_notifications.value.contains(alert)) _notifications.value = _notifications.value + alert
        }
        if (temperature.value > 85f) {
            val alert = "PROACTIVE ALERT: Thermal runaway risk on ${activeTire.value.label}. Reduce speed immediately."
            if (!_notifications.value.contains(alert)) _notifications.value = _notifications.value + alert
        }
    }

    fun clearNotifications() {
        _notifications.value = emptyList()
    }

    // Save current telemetry profile to History (Room) with geographic mapping coordinates support
    fun saveScanToHistory(
        title: String,
        latitude: Double = 36.2704 + (Math.random() - 0.5) * 0.1,
        longitude: Double = -121.8081 + (Math.random() - 0.5) * 0.1,
        routeName: String = "Pacific Coast Hwy",
        onSuccess: () -> Unit = {}
    ) {
        viewModelScope.launch {
            val hScore = healthScore.value
            val notesCombined = _analysisState.value?.let {
                "Analysis: ${it.analysis}\n\nSafety: ${it.safety}\n\nTimeline: ${it.timeline}\n\nRemainingLife: ${it.remainingLifePrediction}"
            } ?: "Diagnostic logs standard check"

            val scan = TireScan(
                timestamp = System.currentTimeMillis(),
                title = title,
                speed = speed.value,
                pressure = pressure.value,
                temperature = temperature.value,
                wearPattern = wearPattern.value,
                overallHealth = hScore,
                notes = notesCombined,
                latitude = latitude,
                longitude = longitude,
                routeName = routeName
            )
            repository.insert(scan)
            onSuccess()
        }
    }

    // Restores a past scan parameters to study tire's historical state
    fun loadHistoricalScan(scan: TireScan) {
        speed.value = scan.speed
        pressure.value = scan.pressure
        temperature.value = scan.temperature
        wearPattern.value = scan.wearPattern
        
        val parsedAnalysis = scan.notes.substringBefore("\n\nSafety:").removePrefix("Analysis: ").trim()
        val parsedSafety = scan.notes.substringAfter("Safety: ").substringBefore("\n\nTimeline:").trim()
        val parsedTimeline = if (scan.notes.contains("\n\nRemainingLife:")) {
            scan.notes.substringAfter("Timeline: ").substringBefore("\n\nRemainingLife:").trim()
        } else {
            scan.notes.substringAfter("Timeline: ").trim()
        }
        val parsedRemainingLife = if (scan.notes.contains("\n\nRemainingLife:")) {
            scan.notes.substringAfter("RemainingLife: ").trim()
        } else {
            "35,000 km (21,700 miles) - historic log average"
        }

        _analysisState.value = GeminiAnalysis(
            analysis = parsedAnalysis,
            safety = parsedSafety,
            timeline = parsedTimeline,
            remainingLifePrediction = parsedRemainingLife
        )
    }

    fun deleteScan(id: Long) {
        viewModelScope.launch {
            repository.deleteById(id)
        }
    }

    fun seedDemoData() {
        viewModelScope.launch {
            val currTime = System.currentTimeMillis()
            val dayMs = 24L * 60 * 60 * 1000

            val logs = listOf(
                TireScan(
                    timestamp = currTime - 28 * dayMs,
                    title = "Baseline Calibration Scan",
                    speed = 50f,
                    pressure = 34.0f,
                    temperature = 28f,
                    wearPattern = "Normal",
                    overallHealth = 98,
                    notes = "Analysis: Optimal contact pressure, balanced wear pattern. Baseline calibration is nominal. \n\nSafety: Low blowout risk, operating fully within parameters.\n\nTimeline: • Track PSI monthly.",
                    latitude = 36.2704, // Big Sur Vista Point
                    longitude = -121.8081,
                    routeName = "Pacific Coast Hwy"
                ),
                TireScan(
                    timestamp = currTime - 21 * dayMs,
                    title = "Highway Cruise High Temp",
                    speed = 100f,
                    pressure = 32.5f,
                    temperature = 44f,
                    wearPattern = "Normal",
                    overallHealth = 94,
                    notes = "Analysis: Tire temperature elevated slightly due to continuous high speed. Inflation pressure remains within safe boundaries.\n\nSafety: Nominal status.\n\nTimeline: • Next scheduled check in 14 days.",
                    latitude = 35.6852, // San Simeon
                    longitude = -121.1667,
                    routeName = "Pacific Coast Hwy"
                ),
                TireScan(
                    timestamp = currTime - 14 * dayMs,
                    title = "Thermal Stress Heat Peak",
                    speed = 85f,
                    pressure = 30.2f,
                    temperature = 68f,
                    wearPattern = "Center Wear",
                    overallHealth = 81,
                    notes = "Analysis: Excessive high temperature on center tread. High friction tarmac is accelerating localized rubber wear.\n\nSafety: Medium safety concern. Allow tires to cool down.\n\nTimeline: • Check temperature thresholds.",
                    latitude = 36.4622, // Death Valley Furnace Creek
                    longitude = -116.8656,
                    routeName = "Death Valley Extreme"
                ),
                TireScan(
                    timestamp = currTime - 7 * dayMs,
                    title = "Critical Mountain Descent",
                    speed = 75f,
                    pressure = 24.2f,
                    temperature = 58f,
                    wearPattern = "Edge Wear",
                    overallHealth = 71,
                    notes = "Analysis: Low pressure causing tire shoulders to bear disproportionate loads, accelerating thermal buildup and friction.\n\nSafety: Heat levels warning. Avoid speed cornering.\n\nTimeline: • Swap axels if edge wear worsens.",
                    latitude = 36.2411, // Death Valley Badwater Basin
                    longitude = -116.8258,
                    routeName = "Death Valley Extreme"
                ),
                TireScan(
                    timestamp = currTime - 1 * dayMs,
                    title = "Urban Start-Stop Deflation",
                    speed = 45f,
                    pressure = 19.5f,
                    temperature = 39f,
                    wearPattern = "Camber Wear",
                    overallHealth = 48,
                    notes = "Analysis: Dangerous under-inflation coupled with high thermal loading. Sidewall flex generates critical friction under speed.\n\nSafety: WARNING: High temperature blowout risk at highway speeds.\n\nTimeline: • Perform wheel alignment. • Swap or rotate tires immediately.",
                    latitude = 37.3382, // San Jose
                    longitude = -121.8863,
                    routeName = "Silicon Valley Commute"
                )
            )

            logs.forEach { repository.insert(it) }
        }
    }
}
