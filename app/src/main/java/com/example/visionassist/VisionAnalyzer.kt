package com.example.visionassist

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.os.SystemClock
import android.util.Half
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.EnumSet
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
//import androidx.core.graphics.scale
import androidx.core.graphics.createBitmap

data class NavState(
    val command: String,
    val sectorScores: FloatArray,
    val bestSector: Int,
    val frameWidth: Float,
    val frameHeight: Float,
    val aStarPath: List<PointF>? = null,
    val spatialRelations: List<String> = emptyList() // FIX: Added spatialRelations property
)

//@SuppressLint("HalfFloat")
class VisionAnalyzer(
    private val context: Context,
    private val spatialAnalyzer: SpatialAnalyzer,
    private val onSceneProcessed: (List<DetectedObject>, NavState, Boolean, String?) -> Unit
) : ImageAnalysis.Analyzer {

    var activeTargets: List<String> = emptyList()

    private val cocoLabels = mapOf(
        0 to "person",
        56 to "chair",
        57 to "couch",
        58 to "potted plant",
        59 to "bed",
        60 to "dining table",
        62 to "tv",
        66 to "keyboard",
        67 to "cell phone",
        73 to "book"
    )

    private var handLandmarker: HandLandmarker? = null
    private var yoloInterpreter: Interpreter? = null
    private var ortEnv: OrtEnvironment? = null
    private var depthSession: OrtSession? = null

    private var lastAnalyzeTime = 0L
    private val frameIntervalMs = 200L // 5FPS

    private val commandHistory = mutableListOf<String>()
    private val detectedObjects = mutableListOf<DetectedObject>()
    private val yoloInputSize = 320
    private val depthInputSize = 224
    private val depthIntValues = IntArray(depthInputSize * depthInputSize)

    // 1. Pre-allocate the ONNX Input Buffer
    private val depthFp16Buffer by lazy {
        ByteBuffer.allocateDirect(1 * 3 * depthInputSize * depthInputSize * 2)
            .order(ByteOrder.nativeOrder())
    }

    // 2. Pre-allocate the 2D Output Map
    val depthOutputMap = Array(depthInputSize) { FloatArray(depthInputSize) }

    // Pre-calculated normalization constants
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile private var latestPointerX: Float? = null
    @Volatile private var latestPointerY: Float? = null

    private lateinit var imageProcessor: ImageProcessor
    private lateinit var tensorImage: TensorImage

    // 3. Pre-allocated Output Buffer for YOLOv11
    // Hold the raw byte memory
    private val yoloByteBuffer by lazy {
        ByteBuffer.allocateDirect(1 * 84 * 2100 * 4).order(ByteOrder.nativeOrder())
    }
    // Create a Float view over that exact same memory block
    private val yoloFloatBuffer by lazy {
        yoloByteBuffer.asFloatBuffer()
    }

    private val gridW = 40
    private val gridH = 30
    private val costMap = Array(gridH) { FloatArray(gridW) }
    private val gScore = Array(gridH) { FloatArray(gridW) }
    private val cameFrom = Array(gridH) { IntArray(gridW) }

    private val dirs = arrayOf(
        intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0),
        intArrayOf(1, 1), intArrayOf(-1, -1), intArrayOf(1, -1), intArrayOf(-1, 1)
    )
    private val depthScaledBitmap by lazy { createBitmap(depthInputSize, depthInputSize) }
    private val depthCanvas by lazy { android.graphics.Canvas(depthScaledBitmap) }
    private val depthMatrix = Matrix()

    init {
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            val optionsBuilder = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(2)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, inputImage ->
                    handleHandLandmarks(result, inputImage.width, inputImage.height)
                }
                .setErrorListener { error -> Log.e("VisionAnalyzer", "MediaPipe Async Error: ${error.message}") }

            handLandmarker = HandLandmarker.createFromOptions(context, optionsBuilder.build())
        } catch (e: Exception) { Log.e("VisionAnalyzer", "Error creating mediapipe hand landmarker", e) }

        try {
            val modelBuffer = loadModelFile("yolo11n_int8.tflite")
            modelBuffer.order(ByteOrder.nativeOrder())

            val options = Interpreter.Options().apply {
                setNumThreads(2)
                useNNAPI = true
            }

            yoloInterpreter = Interpreter(modelBuffer, options)

            val yoloInputType = yoloInterpreter!!.getInputTensor(0).dataType()
            tensorImage = TensorImage(yoloInputType)

            val processorBuilder = ImageProcessor.Builder().add(ResizeOp(yoloInputSize, yoloInputSize, ResizeOp.ResizeMethod.BILINEAR))
            if (yoloInputType == DataType.FLOAT32) processorBuilder.add(NormalizeOp(0f, 255f))
            imageProcessor = processorBuilder.build()

        } catch (e: Exception) { Log.e("VisionAnalyzer", "Error creating yolo interpreter", e) }

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                try {
                    addConfigEntry("session.execution_mode", "ORT_SEQUENTIAL")
                    val qnnOptions = mutableMapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "high_performance", "precision" to "fp16")
                    qnnOptions.forEach { (key, value) -> addConfigEntry("ep.qnn.$key", value) }
                } catch (e: Exception) {
                    try { addNnapi(EnumSet.of(NNAPIFlags.USE_FP16)) } catch (e2: Exception) { Log.e("VisionAnalyzer", "Error adding NNApi config", e2) }
                }
                addConfigEntry("session.use_memory_pattern", "1")
            }

            val modelFile = File(context.cacheDir, "depth_model.onnx")
            if (!modelFile.exists()) {
                context.assets.open("depth_anything_v2_fp16.onnx").use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            depthSession = ortEnv?.createSession(modelFile.absolutePath, sessionOptions)
        } catch (e: Exception) { Log.e("VisionAnalyzer", "Error creating depth session", e) }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fd = context.assets.openFd(modelName)
        return FileInputStream(fd.fileDescriptor).use { inputStream ->
            inputStream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    private fun handleHandLandmarks(result: HandLandmarkerResult, width: Int, height: Int) {
        if (result.landmarks().isEmpty()) {
            latestPointerX = null; latestPointerY = null
            return
        }
        val indexFinger = result.landmarks()[0][8]
        latestPointerX = indexFinger.x() * width
        latestPointerY = indexFinger.y() * height
    }

//    @SuppressLint("HalfFloat")
    @SuppressLint("HalfFloat")
    @ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        if (currentTime - lastAnalyzeTime < frameIntervalMs) {
            imageProxy.close()
            return
        }

        lastAnalyzeTime = currentTime
        detectedObjects.clear()

        val rawBitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val bitmap = rotateBitmap(rawBitmap, rotationDegrees)

        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()

        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker?.detectAsync(mpImage, SystemClock.uptimeMillis())

        if (ortEnv != null && depthSession != null) {
            try {
                val inputName = depthSession!!.inputNames.iterator().next()
                // Calculate the scale ratio for the hardware canvas
                val scaleX = depthInputSize.toFloat() / bitmap.width
                val scaleY = depthInputSize.toFloat() / bitmap.height
                depthMatrix.setScale(scaleX, scaleY)

// Draw the rotated camera frame onto our pre-allocated depth canvas
                depthCanvas.drawBitmap(bitmap, depthMatrix, null)

// Extract the pixels from the pre-allocated bitmap
                depthScaledBitmap.getPixels(depthIntValues, 0, depthInputSize, 0, 0, depthInputSize, depthInputSize)
                depthFp16Buffer.rewind()

                for (c in 0 until 3) {
                    for (i in 0 until depthInputSize * depthInputSize) {
                        val pixel = depthIntValues[i]
                        val channelVal: Float = when (c) {
                            0 -> ((pixel shr 16) and 0xFF) / 255.0f
                            1 -> ((pixel shr 8) and 0xFF) / 255.0f
                            else -> (pixel and 0xFF) / 255.0f
                        }
                        depthFp16Buffer.putShort(Half.toHalf((channelVal - mean[c]) / std[c]).toInt().toShort())
                    }
                }
                depthFp16Buffer.rewind()

                val shape = longArrayOf(1, 3, depthInputSize.toLong(), depthInputSize.toLong())
                OnnxTensor.createTensor(ortEnv, depthFp16Buffer, shape, OnnxJavaType.FLOAT16).use { tensor ->
                    depthSession!!.run(mapOf(inputName to tensor)).use { result ->
                        val outputTensor = result.get(0) as OnnxTensor
                        val isFp16 = outputTensor.info.type == OnnxJavaType.FLOAT16
                        var minD = Float.MAX_VALUE; var maxD = -Float.MAX_VALUE

                        if (isFp16) {
                            val shortBuffer = outputTensor.byteBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
                            for (y in 0 until depthInputSize) {
                                for (x in 0 until depthInputSize) {
                                    val v = Half.toFloat(shortBuffer.get())
                                    if (v < minD) minD = v; if (v > maxD) maxD = v
                                    depthOutputMap[y][x] = v
                                }
                            }
                        } else {
                            val floatBuffer = outputTensor.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
                            for (y in 0 until depthInputSize) {
                                for (x in 0 until depthInputSize) {
                                    val v = floatBuffer.get()
                                    if (v < minD) minD = v; if (v > maxD) maxD = v
                                    depthOutputMap[y][x] = v
                                }
                            }
                        }

                        val range = maxD - minD
                        if (range > 0) {
                            for (y in 0 until depthInputSize) {
                                for (x in 0 until depthInputSize) {
                                    depthOutputMap[y][x] = ((depthOutputMap[y][x] - minD) / range) * 255f
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) { Log.e("VisionPipeline", "Depth inference failed", e) }
        }

        yoloInterpreter?.let { interpreter ->
            try {
                tensorImage.load(bitmap)
                val processedImage = imageProcessor.process(tensorImage)

                val outTensor = interpreter.getOutputTensor(0)
                val shape = outTensor.shape()
                val numAnchors = if (shape[1] > shape[2]) shape[1] else shape[2]
                val numClasses = if (shape[1] > shape[2]) shape[2] - 4 else shape[1] - 4
                val isTransposed = shape[1] == numAnchors
                val isOutputInt8 = outTensor.dataType() == DataType.INT8 || outTensor.dataType() == DataType.UINT8

                yoloByteBuffer.rewind()
                interpreter.run(processedImage.buffer, yoloByteBuffer)
                yoloByteBuffer.rewind()

                val scale = if (isOutputInt8) outTensor.quantizationParams().scale else 1.0f
                val zeroPoint = if (isOutputInt8) outTensor.quantizationParams().zeroPoint else 0
                val rawBoxes = mutableListOf<DetectedObject>()

                fun getValue(index: Int): Float {
                    return if (isOutputInt8) {
                        // Reads exactly 1 byte from memory
                        val rawByte = yoloByteBuffer.get(index).toInt()
                        (rawByte - zeroPoint) * scale
                    } else {
                        // Reads 4 bytes as a Float
                        yoloFloatBuffer.get(index)
                    }
                }
                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    var classId = -1

                    for (c in 0 until numClasses) {
                        val idx = if (isTransposed) i * (numClasses + 4) + 4 + c else (4 + c) * numAnchors + i
                        val score = getValue(idx)
                        if (score > maxScore) { maxScore = score; classId = c }
                    }

                    if (maxScore > 0.2f) {
                        val cxIdx = if (isTransposed) i * (numClasses + 4) + 0 else 0 + i
                        val cyIdx = if (isTransposed) i * (numClasses + 4) + 1 else 1 * numAnchors + i
                        val bwIdx = if (isTransposed) i * (numClasses + 4) + 2 else 2 * numAnchors + i
                        val bhIdx = if (isTransposed) i * (numClasses + 4) + 3 else 3 * numAnchors + i

                        var cx = getValue(cxIdx)
                        var cy = getValue(cyIdx)
                        var bw = getValue(bwIdx)
                        var bh = getValue(bhIdx)

                        if (cx > 2f || cy > 2f || bw > 2f || bh > 2f) {
                            cx /= yoloInputSize; cy /= yoloInputSize; bw /= yoloInputSize; bh /= yoloInputSize
                        }

                        val x1 = (cx - bw / 2f) * width
                        val y1 = (cy - bh / 2f) * height
                        val x2 = (cx + bw / 2f) * width
                        val y2 = (cy + bh / 2f) * height

                        var avgDepth = 50f
                        val depthCx = (cx * depthInputSize)
                        val depthCy = (cy * depthInputSize)
                        val depthBoxW = (bw * depthInputSize) * 0.2f
                        val depthBoxH = (bh * depthInputSize) * 0.2f

                        val dx1 = (depthCx - depthBoxW / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dy1 = (depthCy - depthBoxH / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dx2 = (depthCx + depthBoxW / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dy2 = (depthCy + depthBoxH / 2f).toInt().coerceIn(0, depthInputSize - 1)

                        var depthSum = 0f; var count = 0
                        for (y in dy1..dy2 step 2) {
                            for (x in dx1..dx2 step 2) {
                                depthSum += depthOutputMap[y][x]
                                count++
                            }
                        }
                        if (count > 0) avgDepth = depthSum / count

                        val isPointedAt = latestPointerX?.let { px -> latestPointerY?.let { py -> RectF(x1, y1, x2, y2).contains(px, py) } } ?: false
                        rawBoxes.add(DetectedObject(cocoLabels[classId] ?: "object", RectF(x1, y1, x2, y2), avgDepth, isPointedAt))
                    }
                }
                detectedObjects.addAll(applyNMS(rawBoxes, 0.4f))
            } catch (e: Exception) { Log.e("VisionPipeline", "YOLO Pipeline failed", e) }
        }

    // --- PATHFINDING & ROUTING ---
    // --- PATHFINDING & ROUTING ---
    val numSectors = 10
    val sectorScores = FloatArray(numSectors)
    var rawNavCmd = "🛑 STOP"
    var aStarPath: List<PointF>? = null

    val sectorWDepth = depthInputSize.toFloat() / numSectors
    val sectorWScreen = width / numSectors.toFloat()

    // 1. Calculate Raw Sector Scores
    for (i in 0 until numSectors) {
        var depthSum = 0f; var count = 0
        val startX = (i * sectorWDepth).toInt()
        val endX = ((i + 1) * sectorWDepth).toInt()

        // Analyze bottom 60% of the depth map
        for (y in (depthInputSize * 0.4).toInt() until depthInputSize) {
            for (x in startX until endX) {
                depthSum += depthOutputMap[y][x]
                count++
            }
        }

        // MATH: 0 is far, 255 is close.
        // We subtract from 255 so that open space (0) yields a high safety score (255).
        sectorScores[i] = max(0f, 255f - (if (count > 0) depthSum / count else 0f))

        // 2. Apply YOLO Obstacle Penalties
        val sLeft = i * sectorWScreen
        val sRight = (i + 1) * sectorWScreen
        for (obj in detectedObjects) {
            if (obj.bbox.right > sLeft && obj.bbox.left < sRight && obj.bbox.bottom > height * 0.4f) {
                // Metric > 150 means the detected object is physically close
                if (obj.distanceMetric > 150f) sectorScores[i] = min(sectorScores[i], 20f)
            }
        }
    }

    // 3. Wall Blindness Check (Direct center-pixel sampling)
    // If the very center is saturated (near 255) and the variance is flat, it's a wall.
    val maxScoreRaw = sectorScores.maxOrNull() ?: 0f
    val minScoreRaw = sectorScores.minOrNull() ?: 0f
    val avgCenterPixel = (depthOutputMap[depthInputSize/2][depthInputSize/2] + depthOutputMap[depthInputSize/2 + 5][depthInputSize/2]) / 2f
    val isWallBlindness = avgCenterPixel > 245f && (maxScoreRaw - minScoreRaw) < 45f

    // 4. Sliding Window Smoothing (Avoids narrow corners by checking neighbor safety)
    var bestSector = 4
    var maxSmoothedScore = -1f
    for (i in 0 until numSectors) {
        val leftScore = if (i > 0) sectorScores[i - 1] else 0f
        val rightScore = if (i < numSectors - 1) sectorScores[i + 1] else 0f

        // 1-2-1 Convolution: Only move somewhere if the neighbors are also relatively safe
        val smoothedScore = (leftScore * 0.5f) + sectorScores[i] + (rightScore * 0.5f)

        if (smoothedScore > maxSmoothedScore) {
            maxSmoothedScore = smoothedScore
            bestSector = i
        }
    }

    // 5. Global Opening Check (For "Last Resort" navigation)
    val centerIsBlocked = sectorScores[4] < 60f && sectorScores[5] < 60f
    val alternativeSector = sectorScores.indices.maxByOrNull { sectorScores[it] } ?: 4
    val hasAlternative = sectorScores[alternativeSector] > 110f

    // 6. Primary Logic Tree
    if (activeTargets.isNotEmpty()) {
        // Check for Arrival (Closest target has high metric)
        var maxTargetMetric = -1f
        for (obj in detectedObjects) {
            if (activeTargets.any { obj.className.contains(it, ignoreCase = true) }) {
                if (obj.distanceMetric > maxTargetMetric) maxTargetMetric = obj.distanceMetric
            }
        }

        if (maxTargetMetric > 220f) {
            rawNavCmd = "🎯 ARRIVED"
            bestSector = 4; aStarPath = null
        } else {
            aStarPath = calculateAStar(depthOutputMap, detectedObjects, activeTargets, width, height)
            if (aStarPath != null && aStarPath.size > 5) {
                val lookAheadIdx = min(aStarPath.size - 1, 5)
                bestSector = (aStarPath[lookAheadIdx].x / sectorWScreen).toInt().coerceIn(0, numSectors - 1)

                rawNavCmd = when {
                    isWallBlindness -> "🛑 BACK UP"
                    centerIsBlocked && hasAlternative -> "⚠️ PATH BLOCKED: MOVE ${if(alternativeSector < 5) "LEFT" else "RIGHT"}"
                    centerIsBlocked -> "🛑 PATH BLOCKED"
                    else -> mapSectorToCommand(bestSector, isTarget = true)
                }
            } else {
                rawNavCmd = "🔍 Scanning for target..."
            }
        }
    } else {
        // General Exploration Mode
        rawNavCmd = when {
            isWallBlindness -> "🛑 BACK UP"
            centerIsBlocked && hasAlternative -> {
                bestSector = alternativeSector
                "⚠️ BLOCKED: TURN ${if(bestSector < 5) "LEFT" else "RIGHT"}"
            }
            centerIsBlocked -> "🛑 STOP"
            else -> mapSectorToCommand(bestSector, isTarget = false)
        }
    }

    commandHistory.add(rawNavCmd)
    if (commandHistory.size > 3) commandHistory.removeAt(0)
    val smoothedNavCmd = commandHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: rawNavCmd

    val spatialRelations = spatialAnalyzer.calculateRelations(detectedObjects)
    val (isCollision, dangerObj) = spatialAnalyzer.checkReflexCollision(detectedObjects)

    onSceneProcessed(
        detectedObjects.toList(),
        NavState(smoothedNavCmd, sectorScores, bestSector, width, height, aStarPath, spatialRelations),
        isCollision,
        dangerObj
    )
    imageProxy.close()
}

    // Helper to map the 10 sectors into 5 granular voice directions
    private fun mapSectorToCommand(sector: Int, isTarget: Boolean): String {
        val prefix = if (isTarget) "Target" else "Go"
        return when (sector) {
            in 0..1 -> "⬅️ Hard Left"
            in 2..3 -> "↙️ Slightly Left"
            in 4..5 -> "⬆️ $prefix Straight"
            in 6..7 -> "↘️ Slightly Right"
            else -> "➡️ Hard Right"
        }
    }

private fun calculateAStar(depthMap: Array<FloatArray>, boxes: List<DetectedObject>, targets: List<String>, imgW: Float, imgH: Float): List<PointF>? {
    for (y in 0 until gridH) {
        for (x in 0 until gridW) { gScore[y][x] = Float.MAX_VALUE; cameFrom[y][x] = -1 }
    }

    val depthH = depthMap.size
    val depthW = depthMap[0].size

    for (y in 0 until gridH) {
        for (x in 0 until gridW) {
            val dy = (y * depthH.toFloat() / gridH).toInt().coerceIn(0, depthH - 1)
            val dx = (x * depthW.toFloat() / gridW).toInt().coerceIn(0, depthW - 1)
            costMap[y][x] = depthMap[dy][dx] * 2f
        }
    }

    var targetBox: RectF? = null
    for (box in boxes) {
        val gx1 = max(0, (box.bbox.left / imgW * gridW).toInt())
        val gy1 = max(0, (box.bbox.top / imgH * gridH).toInt())
        val gx2 = min(gridW - 1, (box.bbox.right / imgW * gridW).toInt())
        val gy2 = min(gridH - 1, (box.bbox.bottom / imgH * gridH).toInt())

        if (targets.any { box.className.contains(it, ignoreCase = true) }) {
            targetBox = RectF(gx1.toFloat(), gy1.toFloat(), gx2.toFloat(), gy2.toFloat())
            for (y in gy1..gy2) for (x in gx1..gx2) costMap[y][x] = 0f
        } else {
            for (y in gy1..gy2) for (x in gx1..gx2) costMap[y][x] += 1000f
        }
    }

    if (targetBox == null) return null

    val startX = gridW / 2
    val startY = gridH - 1
    val goalX = ((targetBox.left + targetBox.right) / 2).toInt().coerceIn(0, gridW - 1)
    val goalY = targetBox.bottom.toInt().coerceIn(0, gridH - 1)

    val queue = PriorityQueue<Triple<Float, Int, Int>>(compareBy { it.first })
    queue.add(Triple(0f, startX, startY))
    gScore[startY][startX] = 0f

    while (queue.isNotEmpty()) {
        val curr = queue.poll()!!
        val cx = curr.second
        val cy = curr.third

        if ((cx == goalX && cy == goalY) || (abs(cx - goalX) <= 1 && abs(cy - goalY) <= 1)) {
            val path = mutableListOf<PointF>()
            var currX = cx; var currY = cy

            while (currX != startX || currY != startY) {
                path.add(PointF(currX * imgW / gridW, currY * imgH / gridH))
                val parentEncoded = cameFrom[currY][currX]
                currX = parentEncoded % gridW
                currY = parentEncoded / gridW
            }
            path.add(PointF(startX * imgW / gridW, startY * imgH / gridH))
            return path.reversed()
        }

        for (d in dirs) {
            val nx = cx + d[0]
            val ny = cy + d[1]

            if (nx in 0 until gridW && ny in 0 until gridH) {
                val penalty = costMap[ny][nx]
                if (penalty >= 1000f) continue

                val moveCost = if (d[0] != 0 && d[1] != 0) 14.14f else 10f
                val tG = gScore[cy][cx] + moveCost + penalty

                if (tG < gScore[ny][nx]) {
                    cameFrom[ny][nx] = cy * gridW + cx
                    gScore[ny][nx] = tG
                    val dx = nx - goalX
                    val dy = ny - goalY
                    val f = tG + sqrt((dx * dx + dy * dy).toFloat()) * 20f
                    queue.add(Triple(f, nx, ny))
                }
            }
        }
    }
    return null
}

    private fun applyNMS(boxes: List<DetectedObject>, iouThreshold: Float): List<DetectedObject> {
        val sorted = boxes.sortedByDescending { it.distanceMetric }
        val selected = mutableListOf<DetectedObject>()
        for (box in sorted) {
            var select = true
            for (sel in selected) if (box.className == sel.className && calculateIoU(box.bbox, sel.bbox) > iouThreshold) { select = false; break }
            if (select) selected.add(box)
        }
        return selected
    }

    private fun calculateIoU(b1: RectF, b2: RectF): Float {
        val i = max(0f, min(b1.right, b2.right) - max(b1.left, b2.left)) * max(0f, min(b1.bottom, b2.bottom) - max(b1.top, b2.top))
        return i / ((b1.width() * b1.height()) + (b2.width() * b2.height()) - i)
    }

    private fun rotateBitmap(b: Bitmap, d: Int): Bitmap = if (d == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(d.toFloat()) }, true)
}