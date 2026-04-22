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
    var lastAgentSpeakTime by remember { mutableStateOf(0L) }
    var lastNavCommand by remember { mutableStateOf("") }
    var lastNavSpeakTime by remember { mutableStateOf(0L) }
    var wasTargetFound by remember { mutableStateOf(false) }

    // PERFORMANCE OPTIMIZATION: Pre-allocate Paint objects
    val textPaint = remember { android.graphics.Paint().apply { color = android.graphics.Color.WHITE; textSize = 55f; isFakeBoldText = true; setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK) } }
    val bgPaint = remember { android.graphics.Paint().apply { color = "#CC000000".toColorInt() } }
    val navPaint = remember { android.graphics.Paint().apply { textSize = 80f; textAlign = android.graphics.Paint.Align.CENTER; isFakeBoldText = true; setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK) } }
    val tapPaint = remember { android.graphics.Paint().apply { color = android.graphics.Color.WHITE; textSize = 45f; textAlign = android.graphics.Paint.Align.CENTER; setShadowLayer(5f, 0f, 0f, android.graphics.Color.BLACK) } }

    val voiceAgent = remember {
        var agent: VoiceAgent? = null
        agent = VoiceAgent(context) { cmd ->
            val action = ruleBasedBrain.processCommand(cmd)

            // Update state
            activeTargets = action.targetObjects
            visionAnalyzerRef?.activeTargets = action.targetObjects
            wasTargetFound = false // Reset pathfinding state on new target

            // Prioritize Agent Voice
            lastAgentSpeakTime = System.currentTimeMillis()
            agent?.speak(action.voiceResponse, force = true)
        }
        agent!!
    }

    // --- STRICT NAVIGATION DEBOUNCER ---
    val navCommand = currentNavState?.command
    LaunchedEffect(navCommand) {
        if (navCommand != null && !navCommand.contains("Scanning")) {
            val currentTime = System.currentTimeMillis()
            val isNewCommand = navCommand != lastNavCommand

            // 1. Agent Priority: Don't talk over Caution/Target alerts for 3.5 seconds
            if (currentTime - lastAgentSpeakTime > 3500L) {

                // 2. Command Gap Enforcement:
                // New directions wait 1.5s to prevent self-interruption.
                // Identical repeating directions wait 4.0s to prevent spam.
                val requiredCooldown = if (isNewCommand) 1500L else 4000L

                if (currentTime - lastNavSpeakTime > requiredCooldown) {
                    val cleanCommand = navCommand.replace(Regex("[^a-zA-Z0-9 ,.]"), "").trim()
                    if (cleanCommand.isNotEmpty()) {
                        // FORCE=TRUE guarantees we flush the audio queue so commands never stack up
                        voiceAgent.speak(cleanCommand, force = true)
                        lastNavCommand = navCommand
                        lastNavSpeakTime = currentTime
                    }
                }
            }
        }
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

                        // --- COLLISION TRIGGER ---
                        if (isCollision && currentTime - lastAgentSpeakTime > 3500L) {
                            lastAgentSpeakTime = currentTime
                            voiceAgent.speak("Caution, $dangerObj close ahead.", force = true)
                        }

                        // --- TARGET FOUND TRIGGER ---
                        val isTargetFound = nav.aStarPath != null && activeTargets.isNotEmpty()
                        if (isTargetFound && !wasTargetFound) {
                            if (currentTime - lastAgentSpeakTime > 2000L) {
                                lastAgentSpeakTime = currentTime
                                voiceAgent.speak("Path to ${activeTargets[0]} found. Follow the directions.", force = true)
                            }
                        } else if (!isTargetFound && wasTargetFound && activeTargets.isNotEmpty()) {
                            if (currentTime - lastAgentSpeakTime > 2000L) {
                                lastAgentSpeakTime = currentTime
                                voiceAgent.speak("Lost target. Scanning.", force = true)
                            }
                        }
                        wasTargetFound = isTargetFound

                        try {
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
                        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
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
                val color = if (i == nav.bestSector) Color.Green else if (s < 40) Color.Red else Color.Gray
                val alpha = if (i == nav.bestSector) 0.3f else 0.1f
                drawRect(color = color.copy(alpha = alpha), topLeft = Offset(i * sectorWidth, size.height * 0.45f), size = Size(sectorWidth, size.height * 0.55f))
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
    }
}