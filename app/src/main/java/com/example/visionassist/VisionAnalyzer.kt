package com.example.visionassist

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

import org.tensorflow.lite.Interpreter
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxJavaType
import android.util.Half

import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.PriorityQueue
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class NavState(
    val command: String,
    val sectorScores: FloatArray,
    val bestSector: Int,
    val frameWidth: Float,
    val frameHeight: Float,
    val aStarPath: List<PointF>? = null
)

@SuppressLint("HalfFloat")
class VisionAnalyzer(
    private val context: Context,
    private val spatialAnalyzer: SpatialAnalyzer,
    private val onSceneProcessed: (List<DetectedObject>, NavState, Boolean, String?) -> Unit
) : ImageAnalysis.Analyzer {

    var activeTargets: List<String> = emptyList()

    private var handLandmarker: HandLandmarker? = null
    private var yoloInterpreter: Interpreter? = null
    private var ortEnv: OrtEnvironment? = null
    private var depthSession: OrtSession? = null

    private var lastAnalyzeTime = 0L
    private val frameIntervalMs = 200L // 5 FPS

    private val commandHistory = mutableListOf<String>()

    private val yoloInputSize = 320
    private val depthInputSize = 224

    private val cocoLabels = mapOf(0 to "person", 56 to "chair", 57 to "couch", 58 to "potted plant", 59 to "bed", 60 to "dining table", 62 to "tv", 66 to "keyboard", 67 to "cell phone", 73 to "book")

    init {
        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
            handLandmarker = HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder().setBaseOptions(baseOptions).setNumHands(2).build())
        } catch (e: Exception) { Log.e("VisionPipeline", "MediaPipe Init Failed.", e) }

        try {
            yoloInterpreter = Interpreter(loadModelFile("yolo11n_int8.tflite"), Interpreter.Options().apply { setNumThreads(4) })
        } catch (e: Exception) { Log.e("VisionPipeline", "YOLO Init Failed.", e) }

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val depthBytes = context.assets.open("depth_anything_v2_fp16.onnx").readBytes()
            depthSession = ortEnv?.createSession(depthBytes, OrtSession.SessionOptions().apply { addNnapi() })
        } catch (e: Exception) { Log.e("VisionPipeline", "Depth Anything Init Failed.", e) }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fd = context.assets.openFd(modelName)
        return FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(image: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastAnalyzeTime < frameIntervalMs || image.image == null) {
            image.close()
            return
        }
        lastAnalyzeTime = currentTime

        val bitmap = image.toBitmap()
        val rotatedBitmap = rotateBitmap(bitmap, image.imageInfo.rotationDegrees)
        val height = rotatedBitmap.height.toFloat()
        val width = rotatedBitmap.width.toFloat()

        val detectedObjects = mutableListOf<DetectedObject>()
        var pointerX: Float? = null
        var pointerY: Float? = null

        handLandmarker?.detect(BitmapImageBuilder(rotatedBitmap).build())?.landmarks()?.forEach { landmarks ->
            pointerX = landmarks[8].x() * width
            pointerY = landmarks[8].y() * height
        }

        var depthOutputMap: Array<FloatArray>? = null
        val env = ortEnv
        val session = depthSession

        if (env != null && session != null) {
            try {
                val inputName = session.inputNames.iterator().next()
                val depthBitmap = Bitmap.createScaledBitmap(rotatedBitmap, depthInputSize, depthInputSize, true)

                val bufferSize = 1 * 3 * depthInputSize * depthInputSize * 2
                val fp16Buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())

                val intValues = IntArray(depthInputSize * depthInputSize)
                depthBitmap.getPixels(intValues, 0, depthBitmap.width, 0, 0, depthBitmap.width, depthBitmap.height)

                val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
                val std = floatArrayOf(0.229f, 0.224f, 0.225f)

                for (c in 0 until 3) {
                    for (i in 0 until depthInputSize * depthInputSize) {
                        val pixel = intValues[i]
                        val channelVal: Float = when (c) {
                            0 -> ((pixel shr 16) and 0xFF) / 255.0f
                            1 -> ((pixel shr 8) and 0xFF) / 255.0f
                            else -> (pixel and 0xFF) / 255.0f
                        }
                        fp16Buffer.putShort(Half.toHalf((channelVal - mean[c]) / std[c]))
                    }
                }
                fp16Buffer.rewind()

                val shape = longArrayOf(1, 3, depthInputSize.toLong(), depthInputSize.toLong())
                OnnxTensor.createTensor(env, fp16Buffer, shape, OnnxJavaType.FLOAT16).use { tensor ->
                    session.run(mapOf(inputName to tensor)).use { result ->
                        val outputTensor = result.get(0) as OnnxTensor
                        val isFp16 = outputTensor.info.type == OnnxJavaType.FLOAT16

                        var minD = Float.MAX_VALUE
                        var maxD = -Float.MAX_VALUE

                        if (isFp16) {
                            val shortBuffer = outputTensor.byteBuffer.order(ByteOrder.nativeOrder()).asShortBuffer()
                            depthOutputMap = Array(depthInputSize) {
                                val row = FloatArray(depthInputSize)
                                for (x in 0 until depthInputSize) {
                                    val v = Half.toFloat(shortBuffer.get())
                                    if (v < minD) minD = v
                                    if (v > maxD) maxD = v
                                    row[x] = v
                                }
                                row
                            }
                        } else {
                            val floatBuffer = outputTensor.byteBuffer.order(ByteOrder.nativeOrder()).asFloatBuffer()
                            depthOutputMap = Array(depthInputSize) {
                                val row = FloatArray(depthInputSize)
                                for (x in 0 until depthInputSize) {
                                    val v = floatBuffer.get()
                                    if (v < minD) minD = v
                                    if (v > maxD) maxD = v
                                    row[x] = v
                                }
                                row
                            }
                        }

                        val range = maxD - minD
                        if (range > 0 && depthOutputMap != null) {
                            for (y in 0 until depthInputSize) {
                                for (x in 0 until depthInputSize) {
                                    depthOutputMap!![y][x] = ((depthOutputMap!![y][x] - minD) / range) * 255f
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) { Log.e("VisionPipeline", "Depth inference failed", e) }
        }

        // --- BULLETPROOF YOLO PARSER ---
        yoloInterpreter?.let { interpreter ->
            try {
                val scaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, yoloInputSize, yoloInputSize, true)
                val pixels = IntArray(yoloInputSize * yoloInputSize)
                scaledBitmap.getPixels(pixels, 0, yoloInputSize, 0, 0, yoloInputSize, yoloInputSize)

                val inputTensor = interpreter.getInputTensor(0)
                val isInt8Input = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8 || inputTensor.dataType() == org.tensorflow.lite.DataType.INT8

                val inputBuffer = if (isInt8Input) {
                    val buffer = ByteBuffer.allocateDirect(yoloInputSize * yoloInputSize * 3).order(ByteOrder.nativeOrder())
                    for (pixel in pixels) {
                        buffer.put(((pixel shr 16) and 0xFF).toByte())
                        buffer.put(((pixel shr 8) and 0xFF).toByte())
                        buffer.put((pixel and 0xFF).toByte())
                    }
                    buffer
                } else {
                    val buffer = ByteBuffer.allocateDirect(yoloInputSize * yoloInputSize * 3 * 4).order(ByteOrder.nativeOrder())
                    for (pixel in pixels) {
                        buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
                        buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
                        buffer.putFloat((pixel and 0xFF) / 255.0f)
                    }
                    buffer
                }
                inputBuffer.rewind()

                val outTensor = interpreter.getOutputTensor(0)
                val shape = outTensor.shape()
                val numAnchors = if (shape[1] > shape[2]) shape[1] else shape[2]
                val numClasses = if (shape[1] > shape[2]) shape[2] - 4 else shape[1] - 4
                val isTransposed = shape[1] == numAnchors

                val outputBuffer = ByteBuffer.allocateDirect(shape[1] * shape[2] * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

                interpreter.run(inputBuffer, outputBuffer)
                outputBuffer.rewind()

                val rawBoxes = mutableListOf<DetectedObject>()
                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    var classId = -1

                    for (c in 0 until numClasses) {
                        val score = if (isTransposed) outputBuffer.get(i * (numClasses + 4) + 4 + c) else outputBuffer.get((4 + c) * numAnchors + i)
                        if (score > maxScore) { maxScore = score; classId = c }
                    }

                    if (maxScore > 0.2f) {
                        val cx = if (isTransposed) outputBuffer.get(i * (numClasses + 4) + 0) else outputBuffer.get(0 * numAnchors + i)
                        val cy = if (isTransposed) outputBuffer.get(i * (numClasses + 4) + 1) else outputBuffer.get(1 * numAnchors + i)
                        val bw = if (isTransposed) outputBuffer.get(i * (numClasses + 4) + 2) else outputBuffer.get(2 * numAnchors + i)
                        val bh = if (isTransposed) outputBuffer.get(i * (numClasses + 4) + 3) else outputBuffer.get(3 * numAnchors + i)

                        val x1 = (cx - bw / 2f) / yoloInputSize * width
                        val y1 = (cy - bh / 2f) / yoloInputSize * height
                        val x2 = (cx + bw / 2f) / yoloInputSize * width
                        val y2 = (cy + bh / 2f) / yoloInputSize * height

                        var avgDepth = 50f
                        if (depthOutputMap != null) {
                            val dx1 = ((x1 / width) * depthInputSize).toInt().coerceIn(0, depthInputSize - 1)
                            val dy1 = ((y1 / height) * depthInputSize).toInt().coerceIn(0, depthInputSize - 1)
                            val dx2 = ((x2 / width) * depthInputSize).toInt().coerceIn(0, depthInputSize - 1)
                            val dy2 = ((y2 / height) * depthInputSize).toInt().coerceIn(0, depthInputSize - 1)
                            var depthSum = 0f; var count = 0
                            for (y in dy1..dy2) for (x in dx1..dx2) { depthSum += depthOutputMap!![y][x]; count++ }
                            if (count > 0) avgDepth = depthSum / count
                        }
                        rawBoxes.add(DetectedObject(cocoLabels[classId] ?: "object", RectF(x1, y1, x2, y2), avgDepth, pointerX?.let { px -> pointerY?.let { py -> RectF(x1, y1, x2, y2).contains(px, py) } } ?: false))
                    }
                }
                detectedObjects.addAll(applyNMS(rawBoxes, 0.4f))
            } catch (e: Exception) { Log.e("VisionPipeline", "YOLO failed", e) }
        }

        // --- PATHFINDING & ROUTING ---
        val numSectors = 10
        val sectorScores = FloatArray(numSectors)
        var bestSector = -1
        var maxScore = -1f
        var rawNavCmd = "🛑 STOP"

        var aStarPath: List<PointF>? = null

        if (depthOutputMap != null) {
            val sectorWMap = depthInputSize / numSectors
            for (i in 0 until numSectors) {
                var depthSum = 0f; var count = 0
                for (y in (depthInputSize * 0.4).toInt() until depthInputSize) {
                    for (x in i * sectorWMap until (i + 1) * sectorWMap) { depthSum += depthOutputMap!![y][x]; count++ }
                }
                sectorScores[i] = max(0f, 255f - (if (count > 0) depthSum / count else 0f))

                val sLeft = i * (width / numSectors)
                val sRight = (i + 1) * (width / numSectors)
                for (obj in detectedObjects) {
                    if (obj.bbox.right > sLeft && obj.bbox.left < sRight && obj.bbox.bottom > height * 0.4f) {
                        if (obj.distanceMetric > 150f) sectorScores[i] -= 100f
                    }
                }
                sectorScores[i] = max(0f, sectorScores[i])

                if (sectorScores[i] > maxScore) { maxScore = sectorScores[i]; bestSector = i }
            }

            // Reliable STOP Check: Look at the immediate center bottom for major obstructions.
            val centerObstruction = sectorScores[4] < 60f && sectorScores[5] < 60f

            if (activeTargets.isNotEmpty()) {
                aStarPath = calculateAStar(depthOutputMap!!, detectedObjects, activeTargets, width, height)

                if (aStarPath != null && aStarPath.size > 5) {
                    // FIX 1: Path is now Ordered [Start -> Goal]. Look 5 nodes ahead of the User's feet.
                    val lookAheadIdx = min(aStarPath.size - 1, 5)
                    val lookAheadPoint = aStarPath[lookAheadIdx]

                    val targetSector = (lookAheadPoint.x / (width / numSectors)).toInt().coerceIn(0, numSectors - 1)
                    bestSector = targetSector

                    rawNavCmd = when {
                        centerObstruction && maxScore < 40f -> "🛑 PATH BLOCKED"
                        bestSector < 4 -> "⬅️ Target Left"
                        bestSector > 5 -> "➡️ Target Right"
                        else -> "⬆️ Target Straight"
                    }
                } else {
                    rawNavCmd = "🔍 Scanning for target..."
                }
            } else {
                rawNavCmd = when {
                    centerObstruction || maxScore < 40f -> "🛑 STOP" // FIX 2: Increased reliability threshold
                    bestSector < 4 -> "⬅️ Turn Left"
                    bestSector > 5 -> "➡️ Turn Right"
                    else -> "⬆️ Go Straight"
                }
            }
        }

        commandHistory.add(rawNavCmd)
        if (commandHistory.size > 3) commandHistory.removeAt(0)
        val smoothedNavCmd = commandHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: rawNavCmd

        val (isCollision, dangerObj) = spatialAnalyzer.checkReflexCollision(detectedObjects)
        onSceneProcessed(detectedObjects, NavState(smoothedNavCmd, sectorScores, bestSector, width, height, aStarPath), isCollision, dangerObj)
        image.close()
    }

    private fun calculateAStar(depthMap: Array<FloatArray>, boxes: List<DetectedObject>, targets: List<String>, imgW: Float, imgH: Float): List<PointF>? {
        val gridW = 40
        val gridH = 30
        val costMap = Array(gridH) { FloatArray(gridW) }

        val depthH = depthMap.size
        val depthW = depthMap[0].size

        // FIX 3: Integrate real Depth Map into A* to break Left-Bias. Darker (further) pixels are inherently cheaper.
        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                val dy = (y * depthH / gridH).coerceIn(0, depthH - 1)
                val dx = (x * depthW / gridW).coerceIn(0, depthW - 1)
                costMap[y][x] = depthMap[dy][dx] * 2f // Scale physical depth penalty
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

        val start = Pair(gridW / 2, gridH - 1)
        val goal = Pair(((targetBox.left + targetBox.right)/2).toInt(), targetBox.bottom.toInt().coerceIn(0, gridH - 1))

        val queue = PriorityQueue<Triple<Float, Int, Pair<Int, Int>>>(compareBy { it.first })
        queue.add(Triple(0f, 0, start))
        val cameFrom = mutableMapOf<Pair<Int, Int>, Pair<Int, Int>>()
        val gScore = mutableMapOf(start to 0f)
        val dirs = listOf(Pair(0,1), Pair(0,-1), Pair(1,0), Pair(-1,0), Pair(1,1), Pair(-1,-1), Pair(1,-1), Pair(-1,1))

        while(queue.isNotEmpty()) {
            val curr = queue.poll()!!.third
            if (curr == goal || (abs(curr.first - goal.first) <= 1 && abs(curr.second - goal.second) <= 1)) {
                val path = mutableListOf<PointF>()
                var step = curr
                while(step != start) {
                    path.add(PointF(step.first * imgW / gridW, step.second * imgH / gridH))
                    step = cameFrom[step]!!
                }
                path.add(PointF(start.first * imgW / gridW, start.second * imgH / gridH))

                // Return reversed so [0] is the User Start, extending outward to the Goal
                return path.reversed()
            }

            for (d in dirs) {
                val nx = curr.first + d.first
                val ny = curr.second + d.second
                if (nx in 0 until gridW && ny in 0 until gridH) {
                    val penalty = costMap[ny][nx]
                    if (penalty >= 1000f) continue

                    val tG = gScore[curr]!! + sqrt((d.first*d.first + d.second*d.second).toFloat()) * 10f + penalty
                    val neighbor = Pair(nx, ny)
                    if (tG < gScore.getOrDefault(neighbor, Float.MAX_VALUE)) {
                        cameFrom[neighbor] = curr
                        gScore[neighbor] = tG
                        val dx = nx - goal.first
                        val dy = ny - goal.second
                        val f = tG + sqrt((dx * dx + dy * dy).toFloat()) * 20f // Standard A* Heuristic
                        queue.add(Triple(f, 0, neighbor))
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