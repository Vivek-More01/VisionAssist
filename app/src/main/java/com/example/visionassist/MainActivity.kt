package com.example.visionassist

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import androidx.core.graphics.toColorInt

class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestPermissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO))
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    VisionMainScreen()
                }
            }
        }
    }
}

@Composable
fun VisionMainScreen() {
    var isFlashlightOn by remember { mutableStateOf(false) }
    var cameraControl by remember { mutableStateOf<androidx.camera.core.CameraControl?>(null) }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val spatialAnalyzer = remember { SpatialAnalyzer() }
    val ruleBasedBrain = remember { RuleBasedBrain() }

    var renderObjects by remember { mutableStateOf<List<DetectedObject>>(emptyList()) }
    var currentNavState by remember { mutableStateOf<NavState?>(null) }
    var activeTargets by remember { mutableStateOf<List<String>>(emptyList()) }
    var latestSceneJson by remember { mutableStateOf("{}") }

    val currentSceneJson by rememberUpdatedState(latestSceneJson)
    var visionAnalyzerRef by remember { mutableStateOf<VisionAnalyzer?>(null) }

    // TTS Timers & States for Strict Debouncing
    var lastAgentSpeakTime by remember { mutableLongStateOf(0L) }
    var lastNavCommand by remember { mutableStateOf("") }
    var lastNavSpeakTime by remember { mutableLongStateOf(0L) }
    var wasTargetFound by remember { mutableStateOf(false) }

    // PERFORMANCE OPTIMIZATION: Pre-allocate Paint objects
    val textPaint = remember { android.graphics.Paint().apply { color = android.graphics.Color.WHITE; textSize = 55f; isFakeBoldText = true; setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK) } }
    val bgPaint = remember { android.graphics.Paint().apply { color = "#CC000000".toColorInt() } }
    val navPaint = remember { android.graphics.Paint().apply { textSize = 80f; textAlign = android.graphics.Paint.Align.CENTER; isFakeBoldText = true; setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK) } }
    val tapPaint = remember { android.graphics.Paint().apply { color = android.graphics.Color.WHITE; textSize = 45f; textAlign = android.graphics.Paint.Align.CENTER; setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK) } }
    val scorePaint = remember {
        android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 45f
            textAlign = android.graphics.Paint.Align.CENTER
            isFakeBoldText = true
            setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK)
        }
    }
    // Keep your State variables at the top...
    val latestNavState by rememberUpdatedState(currentNavState) // FIX: Ensures VoiceAgent reads fresh data
    var lastManualCommandTime by remember { mutableStateOf(0L) }
    val voiceAgent = remember {
        var agent: VoiceAgent? = null
        agent = VoiceAgent(context) { cmd ->
            val action = ruleBasedBrain.processCommand(cmd)

            if (action.intent == "describe") {
                // Uses 'latestNavState' to ensure the lambda reads fresh Compose state
                val relations = latestNavState?.spatialRelations ?: emptyList()
                val uniqueObjects = renderObjects.map { it.className }.distinct()

                val description = if (relations.isNotEmpty()) {
                    "I see ${uniqueObjects.joinToString(", ")}. " + relations.joinToString(". ")
                } else if (uniqueObjects.isNotEmpty()) {
                    "I see ${uniqueObjects.joinToString(", ")}. No specific interactions detected."
                } else {
                    "The scene is clear."
                }

                // Tell the Priority Queue to say this immediately, interrupting normal navigation
                lastManualCommandTime = System.currentTimeMillis() // START GRACE PERIOD
                agent?.speak(description, priority = SpeechPriority.HIGH)
            } else {
                activeTargets = action.targetObjects
                visionAnalyzerRef?.activeTargets = action.targetObjects
                wasTargetFound = false

                // Acknowledge the user's command immediately
                lastManualCommandTime = System.currentTimeMillis() // START GRACE PERIOD
                agent?.speak(action.voiceResponse, priority = SpeechPriority.HIGH)
            }
        }
        agent!!
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
            voiceAgent.shutdown()
        }
    }

    Box(modifier = Modifier.fillMaxSize().clickable {
        Toast.makeText(context, "🎤 Listening...", Toast.LENGTH_SHORT).show()
        voiceAgent.listenForCommand()
    }) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val providerFuture = ProcessCameraProvider.getInstance(ctx)
                providerFuture.addListener({
                    val provider = providerFuture.get()
                    val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }

                    val analyzer = VisionAnalyzer(ctx, spatialAnalyzer) { objs, nav, isCollision, dangerObj ->
                        renderObjects = objs
                        currentNavState = nav
                        val currentTime = System.currentTimeMillis()

                        // --- UNIFIED AUDIO DISPATCHER (Priority Queued) ---

                        val cleanCommand = nav.command.replace(Regex("[^a-zA-Z0-9 ,.]"), "").trim()

                        // If it has been less than 8 seconds since the user asked a question, stay quiet.
                        val isGracePeriod = (currentTime - lastManualCommandTime) < 8000L

                        // 1. COLLISION (Highest Priority - ALWAYS plays, ignores grace period)
                        if (isCollision) {
                            voiceAgent.speak("Caution, $dangerObj close ahead.", SpeechPriority.CRITICAL)
                        }
                        // 2. CRITICAL NAVIGATION OVERRIDES
                        else if (!isGracePeriod && (nav.command.contains("STOP") || nav.command.contains("BACK UP") || nav.command.contains("BLOCKED"))) {
                            voiceAgent.speak(cleanCommand, SpeechPriority.CRITICAL)
                        }
                        // 3. TARGET ARRIVAL
                        else if (nav.command.contains("ARRIVED") && activeTargets.isNotEmpty()) {
                            voiceAgent.speak("You have reached the ${activeTargets[0]}.", SpeechPriority.HIGH)
                            activeTargets = emptyList()
                            visionAnalyzerRef?.activeTargets = emptyList()
                            wasTargetFound = false
                        }
                        // 4. TARGET SEEKING LOGIC
                        else if (activeTargets.isNotEmpty()) {
                            val isTargetFound = nav.aStarPath != null
                            if (isTargetFound && !wasTargetFound) {
                                voiceAgent.speak("Path to ${activeTargets[0]} found. Follow the directions.", SpeechPriority.HIGH)
                            } else if (!isTargetFound && wasTargetFound) {
                                voiceAgent.speak("Lost target. Scanning.", SpeechPriority.HIGH)
                            }
                            wasTargetFound = isTargetFound

                            // Normal nav is muted during grace period
                            if (!isGracePeriod && cleanCommand.isNotEmpty() && !cleanCommand.contains("Scanning")) {
                                voiceAgent.speak(cleanCommand, SpeechPriority.NORMAL)
                            }
                        }
                        // 5. NORMAL EXPLORATION
                        else {
                            // Normal nav is muted during grace period
                            if (!isGracePeriod && cleanCommand.isNotEmpty() && !cleanCommand.contains("Scanning")) {
                                voiceAgent.speak(cleanCommand, SpeechPriority.NORMAL)
                            }
                        }
                        // JSON Construction logic remains the same...
                        try {
                            // ...
                            val json = JSONObject().apply {
                                put("navigation", JSONObject().apply {
                                    put("command", nav.command)
                                    put("bestSector", nav.bestSector)
                                    put("sectorScores", JSONArray(nav.sectorScores.map { it.toDouble() }))
                                })
                                val objArr = JSONArray()
                                objs.forEach { o -> objArr.put(JSONObject().apply { put("label", o.className); put("distance", o.distanceMetric.toDouble()) }) }
                                put("detectedObjects", objArr)
                            }
                            latestSceneJson = json.toString()
                        } catch (e: Exception) { Log.e("Vision", "JSON Err", e) }
                    }

                    visionAnalyzerRef = analyzer
                    val analysis = ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also { it.setAnalyzer(cameraExecutor, analyzer) }

                    try {
                        provider.unbindAll()
                        val camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                        // Capture the control instance!
                        cameraControl = camera.cameraControl
                    } catch (e: Exception) { Log.e("Vision", "Bind failed", e) }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            }
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val nav = currentNavState ?: return@Canvas
            val scale = maxOf(size.width / nav.frameWidth, size.height / nav.frameHeight)
            val oX = (size.width - nav.frameWidth * scale) / 2f
            val oY = (size.height - nav.frameHeight * scale) / 2f

            val sectorWidth = size.width / 10f
            nav.sectorScores.forEachIndexed { i, s ->
                // Draw the shaded rectangle
                val color = if (i == nav.bestSector) Color.Green else if (s < 40) Color.Red else Color.Gray
                val alpha = if (i == nav.bestSector) 0.3f else 0.1f
                drawRect(
                    color = color.copy(alpha = alpha),
                    topLeft = Offset(i * sectorWidth, size.height * 0.45f),
                    size = Size(sectorWidth, size.height * 0.55f)
                )

                // NEW: Draw the exact numeric score at the bottom of the sector
                drawContext.canvas.nativeCanvas.drawText(
                    s.toInt().toString(),
                    (i * sectorWidth) + (sectorWidth / 2f), // Center X of the sector
                    size.height * 0.95f,                    // Y position (near the bottom)
                    scorePaint
                )
            }

            nav.aStarPath?.let { pathPoints ->
                if (pathPoints.isNotEmpty()) {
                    val path = Path()
                    val mappedPoints = pathPoints.map { Offset(it.x * scale + oX, it.y * scale + oY) }
                    path.moveTo(mappedPoints.first().x, mappedPoints.first().y)
                    for (i in 1 until mappedPoints.size) { path.lineTo(mappedPoints[i].x, mappedPoints[i].y) }
                    drawPath(path = path, color = Color.Yellow, style = Stroke(width = 12f, join = StrokeJoin.Round))
                }
            }

            renderObjects.forEach { o ->
                val l = o.bbox.left * scale + oX
                val t = o.bbox.top * scale + oY
                val r = o.bbox.right * scale + oX
                val b = o.bbox.bottom * scale + oY

                val isTarget = activeTargets.any { target -> o.className.contains(target, ignoreCase = true) }
                val boxColor = when {
                    isTarget || o.isPointedAt -> Color.Yellow
                    o.distanceMetric > 150f -> Color.Red
                    else -> Color.Cyan
                }

                drawRect(boxColor, Offset(l, t), Size(r - l, b - t), style = Stroke(8f))

                drawContext.canvas.nativeCanvas.apply {
                    val label = "${o.className.uppercase()} [${o.distanceMetric.toInt()}]"
                    val textWidth = textPaint.measureText(label)

                    drawRect(l, t - 65f, l + textWidth + 20f, t, bgPaint)
                    drawText(label, l + 10f, t - 15f, textPaint)
                }
            }

            val displayCommand = if (activeTargets.isNotEmpty() && nav.aStarPath == null) "SEEKING: ${activeTargets.joinToString(", ").uppercase()}" else nav.command

            drawContext.canvas.nativeCanvas.apply {
                navPaint.color = if ("STOP" in displayCommand) android.graphics.Color.RED else android.graphics.Color.GREEN
                drawText(displayCommand, size.width / 2f, 150f, navPaint)
                drawText("TAP SCREEN TO SPEAK", size.width / 2f, size.height - 80f, tapPaint)
            }
        }
        // Put this inside your main Box{}, below the Canvas{} so it floats on top
        androidx.compose.material3.Button(
            onClick = {
                isFlashlightOn = !isFlashlightOn
                cameraControl?.enableTorch(isFlashlightOn)
            },
            modifier = Modifier.align(androidx.compose.ui.Alignment.TopEnd).padding(16.dp)
        ) {
            androidx.compose.material3.Text(if (isFlashlightOn) "🔦 OFF" else "🔦 ON")
        }
    }
}
//    LaunchedEffect(navCommand) {
//        if (navCommand != null && !navCommand.contains("Scanning")) {
//            val currentTime = System.currentTimeMillis()
//            val isNewCommand = navCommand != lastNavCommand
//
//            // 1. Agent Priority: Don't talk over Caution/Target alerts for 3.5 seconds
//            if (currentTime - lastAgentSpeakTime > 3500L) {
//
//                // 2. Command Gap Enforcement:
//                // New directions wait 1.5s to prevent self-interruption.
//                // Identical repeating directions wait 4.0s to prevent spam.
//                val requiredCooldown = if (isNewCommand) 1500L else 4000L
//
//                if (currentTime - lastNavSpeakTime > requiredCooldown) {
//                    val cleanCommand = navCommand.replace(Regex("[^a-zA-Z0-9 ,.]"), "").trim()
//                    if (cleanCommand.isNotEmpty()) {
//                        // FORCE=TRUE guarantees we flush the audio queue so commands never stack up
//                        voiceAgent.speak(cleanCommand, force = true)
//                        lastNavCommand = navCommand
//                        lastNavSpeakTime = currentTime
//                    }
//                }
//            }
//        }
//    }