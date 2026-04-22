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
import org.tensorflow.lite.support.image.ImageProcessor
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.providers.NNAPIFlags
import android.os.SystemClock
import android.util.Half
import androidx.camera.core.ExperimentalGetImage
import androidx.privacysandbox.tools.core.model.Method
import com.google.ar.core.ImageFormat
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File

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
import java.util.EnumSet

data class NavState(
    val command: String,
    val sectorScores: FloatArray,
    val bestSector: Int,
    val frameWidth: Float,
    val frameHeight: Float,
    val aStarPath: List<PointF>?=null
)
//New Code
@SuppressLint("HalfFloat")
class VisionAnalyzer(
    private val context: Context,
    private val spatialAnalyzer: SpatialAnalyzer,
    private val onSceneProcessed: (List<DetectedObject>, NavState, Boolean,String?) -> Unit
) : ImageAnalysis.Analyzer {
    var activeTargets: List<String> = emptyList()
    private val cocoLabels = mapOf(
        0 to "book",
        1 to "bottole",
        2 to "earphone",
        3 to "glass",
        4 to "headphone",
        5 to "keyboard",
        6 to "laptop",
        7 to "mobile",
        8 to "mouse",
        9 to "pen",
        10 to "penstand",
        11 to "curtain",
        12 to "chair",
        13 to "windows",
        14 to "Whiteboard",
        15 to "Desk",
        16 to "Computer-monitor",
        17 to "door",
        18 to "door_handle",
        19 to "human"
    )
    private var handLandmarker: HandLandmarker? = null
    private var yoloInterpreter: Interpreter? = null
    private var ortEnv: OrtEnvironment? = null
    private var depthSession: OrtSession? = null

    private var lastAnalyzeTime = 0L
    private val frameIntervalMs = 200L //5FPS

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

    // 2. Pre-allocate the 2D Output Map (No longer nullable)
    val depthOutputMap = Array(depthInputSize) { FloatArray(depthInputSize) }

    // Pre-calculated normalization constants to save CPU cycles
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Volatile ensures that if the background thread updates this,
    // the main analyze thread sees the change immediately.
    @Volatile
    private var latestPointerX: Float? = null
    @Volatile
    private var latestPointerY: Float? = null

    // Allocate ONCE
    // 1. TFLite Support Image Processor (Handles Resize & Normalize in C++)
    // Explicitly declare the type as ': ImageProcessor' to fix the delegate error
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var tensorImage: TensorImage

    // 3. Pre-allocated Output Buffer for YOLOv11 (1 batch, 84 attributes, 8400 anchors)
// We allocate this directly in native memory once.
    private val yoloOutputBuffer by lazy {
        ByteBuffer.allocateDirect(1 * 84 * 2100 * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
    }

    private val gridW = 40
    private val gridH = 30
    private val costMap = Array(gridH) { FloatArray(gridW) }
    private val gScore = Array(gridH) { FloatArray(gridW) }

    // Encode parent coordinates as an Int (parentY * gridW + parentX) to avoid Pair objects
    private val cameFrom = Array(gridH) { IntArray(gridW) }

    // Pre-allocate directions as primitive arrays
    private val dirs = arrayOf(
        intArrayOf(0, 1), intArrayOf(0, -1), intArrayOf(1, 0), intArrayOf(-1, 0),
        intArrayOf(1, 1), intArrayOf(-1, -1), intArrayOf(1, -1), intArrayOf(-1, 1)
    )

    init {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("hand_landmarker.task")
                // Optional 2026 Optimization: .setDelegate(Delegate.GPU) to offload from CPU
                .build()

            val optionsBuilder = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(2)
                // 1. CRITICAL: Tell MediaPipe we are streaming continuous frames
                .setRunningMode(RunningMode.LIVE_STREAM)
                // 2. CRITICAL: The callback that runs when the inference is finished
                .setResultListener { result, inputImage ->
                    handleHandLandmarks(result, inputImage.width, inputImage.height)
                }
                // 3. Good Practice: Catch any async errors so they don't silently crash the pipeline
                .setErrorListener { error ->
                    Log.e("VisionAnalyzer", "MediaPipe Async Error: ${error.message}")
                }

            handLandmarker = HandLandmarker.createFromOptions(context, optionsBuilder.build())
            Log.d("VisionAnalyzer", "Async HandLandmarker initialized successfully.")
        } catch (e: Exception) {
            Log.e("VisionAnalyzer", "Error creating mediapipe hand landmarker", e)
        }

        try {
            // 1. Prepare the model buffer with the correct byte order
            val modelBuffer = loadModelFile("lab_model.tflite")
            modelBuffer.order(ByteOrder.nativeOrder()) // This ensures the math is read correctly by the CPU/NPU

            // 2. Configure the Interpreter
            val options = Interpreter.Options().apply {
                setNumThreads(2) // Good choice for 2026; prevents overheating
                useNNAPI = true  // Standard hardware acceleration
            }

            // 3. Initialize
            yoloInterpreter = Interpreter(modelBuffer, options)
            Log.d("VisionAnalyzer", "YOLO Interpreter initialized successfully")

// --- NEW DYNAMIC PRE-PROCESSING LOGIC ---
            val yoloInputType = yoloInterpreter!!.getInputTensor(0).dataType()

// 1. Match the TensorImage to the model's expected input type (FLOAT32 or UINT8)
            tensorImage = TensorImage(yoloInputType)

// 2. Build the processor based on the data type
            val processorBuilder = ImageProcessor.Builder()
                .add(ResizeOp(yoloInputSize, yoloInputSize, ResizeOp.ResizeMethod.BILINEAR))

            // If the model is Float, it needs the 0.0-1.0 normalization
            if (yoloInputType == DataType.FLOAT32) {
                processorBuilder.add(NormalizeOp(0f, 255f))
            }
            // If it is INT8/UINT8, no NormalizeOp is added.
            // The TensorImage will natively handle the 0-255 raw byte casting,
            // perfectly replicating your old bit-shifting logic.

            imageProcessor = processorBuilder.build()
            Log.d("VisionAnalyzer", "YOLO Interpreter initialized successfully")

        } catch (e: Exception) {
            Log.e("VisionAnalyzer", "Error creating yolo interpreter", e)
        }

        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions().apply {
                try {
                    addConfigEntry("session.execution_mode", "ORT_SEQUENTIAL")
                    val qnnOptions = mutableMapOf<String, String>().apply {
                        put("backend_path", "libQnnHtp.so")
                        put("htp_performance_mode", "high_performance")
                        put("precision", "fp16")
                    }

                    // This is the actual method call for custom hardware providers
                    // If the phone isn't Snapdragon, this will throw an exception and go to 'catch'
                    qnnOptions.forEach { (key, value) ->
                        addConfigEntry("ep.qnn.$key", value)
                    }

                    Log.d("Vision", "QNN (Hexagon) optimization enabled.")
                } catch (e: Exception) {
                    Log.e("VisionAnalyzer", "Error adding QNN config", e)
                    try {
                        addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                    } catch (e: Exception) {
                        val xnnpackOptions = mapOf("intra_op_num_threads" to "2")
                        addXnnpack(xnnpackOptions)
                        Log.e("VisionAnalyzer", "Error adding NNApi config", e)
                    }
                }
                addConfigEntry("session.use_memory_pattern", "1")
            }

            val modelFile = File(context.cacheDir, "depth_model.onnx")

            if (!modelFile.exists()) {
                context.assets.open("depth_anything_v2_fp16.onnx").use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            val modelPath = modelFile.absolutePath
            depthSession = ortEnv?.createSession(modelPath, sessionOptions)
        } catch (e: Exception) {
            Log.e("VisionAnalyzer", "Error creating depth session", e)
        }
    }

    fun loadModelFile(modelName: String): MappedByteBuffer {
        val fd = context.assets.openFd(modelName)
        return FileInputStream(fd.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            fileChannel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }
    }

    private fun handleHandLandmarks(result: HandLandmarkerResult, width: Int, height: Int) {
        if (result.landmarks().isEmpty()) {
            // If no hand is found, clear the pointers
            latestPointerX = null
            latestPointerY = null
            return
        }

        // Grab the first detected hand
        val landmarks = result.landmarks()[0]

        // Landmark 8 is the Index Finger Tip
        val indexFinger = landmarks[8]

        // Update the class-level variables
        latestPointerX = indexFinger.x() * width
        latestPointerY = indexFinger.y() * height
    }

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
        //Handlandmarker
        // 1. Get the raw bitmap and the hardware rotation flag
        val rawBitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        // 2. Physically rotate the bitmap to match the screen
        val bitmap = rotateBitmap(rawBitmap, rotationDegrees)

        // 2. CRITICAL FIX: Pull dimensions from the rotated bitmap, NOT the raw ImageProxy
        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()

        // 3. Unify MediaPipe: Give it the exact same rotated bitmap YOLO gets
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker?.detectAsync(mpImage, SystemClock.uptimeMillis())
        //DepthAnythingV2
        if (ortEnv != null && depthSession != null) {
            try {
                val inputName = depthSession!!.inputNames.iterator().next()

                // Note: For absolute peak performance, you'd want to avoid createScaledBitmap
                // and use a Canvas to draw to a pre-allocated bitmap, but this is acceptable for now.
                val depthBitmap =
                    Bitmap.createScaledBitmap(bitmap, depthInputSize, depthInputSize, true)
                depthBitmap.getPixels(
                    depthIntValues,
                    0,
                    depthBitmap.width,
                    0,
                    0,
                    depthBitmap.width,
                    depthBitmap.height
                )

                depthFp16Buffer.rewind()

                // 1. High-Speed Image Normalization
                for (c in 0 until 3) {
                    for (i in 0 until depthInputSize * depthInputSize) {
                        val pixel = depthIntValues[i]
                        val channelVal: Float = when (c) {
                            0 -> ((pixel shr 16) and 0xFF) / 255.0f
                            1 -> ((pixel shr 8) and 0xFF) / 255.0f
                            else -> (pixel and 0xFF) / 255.0f
                        }
                        // Calculate $z = \frac{x - \mu}{\sigma}$
                        depthFp16Buffer.putShort(Half.toHalf((channelVal - mean[c]) / std[c]))
                    }
                }
                depthFp16Buffer.rewind()

                // 2. Run Inference
                val shape = longArrayOf(1, 3, depthInputSize.toLong(), depthInputSize.toLong())
                OnnxTensor.createTensor(ortEnv, depthFp16Buffer, shape, OnnxJavaType.FLOAT16)
                    .use { tensor ->
                        depthSession!!.run(mapOf(inputName to tensor)).use { result ->
                            val outputTensor = result.get(0) as OnnxTensor
                            val isFp16 = outputTensor.info.type == OnnxJavaType.FLOAT16

                            var minD = Float.MAX_VALUE
                            var maxD = -Float.MAX_VALUE

                            // 3. Zero-Allocation Output Parsing
                            if (isFp16) {
                                val shortBuffer =
                                    outputTensor.byteBuffer.order(ByteOrder.nativeOrder())
                                        .asShortBuffer()
                                for (y in 0 until depthInputSize) {
                                    for (x in 0 until depthInputSize) {
                                        val v = Half.toFloat(shortBuffer.get())
                                        if (v < minD) minD = v
                                        if (v > maxD) maxD = v
                                        depthOutputMap[y][x] = v // Overwrite existing array
                                    }
                                }
                            } else {
                                val floatBuffer =
                                    outputTensor.byteBuffer.order(ByteOrder.nativeOrder())
                                        .asFloatBuffer()
                                for (y in 0 until depthInputSize) {
                                    for (x in 0 until depthInputSize) {
                                        val v = floatBuffer.get()
                                        if (v < minD) minD = v
                                        if (v > maxD) maxD = v
                                        depthOutputMap[y][x] = v // Overwrite existing array
                                    }
                                }
                            }

                            // 4. In-Place Normalization (0-255)
                            val range = maxD - minD
                            if (range > 0) {
                                for (y in 0 until depthInputSize) {
                                    for (x in 0 until depthInputSize) {
                                        depthOutputMap[y][x] =
                                            ((depthOutputMap[y][x] - minD) / range) * 255f
                                    }
                                }
                            }
                        }
                    }
            } catch (e: Exception) {
                Log.e("VisionPipeline", "Depth inference failed", e)
            }
        }

        //Yolo v11
        yoloInterpreter?.let { interpreter ->
            try {
                // 1. Hardware-Accelerated Pre-processing
                tensorImage.load(bitmap)
                val processedImage = imageProcessor.process(tensorImage)

                // 2. Prepare Inference variables
                val outTensor = interpreter.getOutputTensor(0)
                val shape = outTensor.shape()
                val numAnchors = if (shape[1] > shape[2]) shape[1] else shape[2]
                val numClasses = if (shape[1] > shape[2]) shape[2] - 4 else shape[1] - 4
                val isTransposed = shape[1] == numAnchors

                val isOutputInt8 = outTensor.dataType() == org.tensorflow.lite.DataType.INT8 || outTensor.dataType() == org.tensorflow.lite.DataType.UINT8

                // 3. Run Inference using the Pre-Allocated Output Buffer
                yoloOutputBuffer.rewind()
                interpreter.run(processedImage.buffer, yoloOutputBuffer)
                yoloOutputBuffer.rewind()

                // Extract Quantization Params (if applicable)
                val scale = if (isOutputInt8) outTensor.quantizationParams().scale else 1.0f
                val zeroPoint = if (isOutputInt8) outTensor.quantizationParams().zeroPoint else 0

                val rawBoxes = mutableListOf<DetectedObject>()

                // Helper function for inline targeted dequantization
                fun getValue(index: Int): Float {
                    return if (isOutputInt8) {
                        // Read as Float buffer, but bit-cast to Int to get the raw byte value out of memory
                        val rawByte = java.lang.Float.floatToRawIntBits(yoloOutputBuffer.get(index)) and 0xFF
                        (rawByte - zeroPoint) * scale
                    } else {
                        yoloOutputBuffer.get(index)
                    }
                }

                // 4. Extract Bounding Boxes
                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    var classId = -1

                    // Only parse the classes we need
                    for (c in 0 until numClasses) {
                        val idx = if (isTransposed) i * (numClasses + 4) + 4 + c else (4 + c) * numAnchors + i
                        val score = getValue(idx)

                        if (score > maxScore) {
                            maxScore = score
                            classId = c
                        }
                    }

                    if (maxScore > 0.2f) { // Confidence Threshold

                        // Only extract coordinate data if the score passes the threshold!
                        // This saves thousands of math operations per frame.
                        val cxIdx = if (isTransposed) i * (numClasses + 4) + 0 else 0 * numAnchors + i
                        val cyIdx = if (isTransposed) i * (numClasses + 4) + 1 else 1 * numAnchors + i
                        val bwIdx = if (isTransposed) i * (numClasses + 4) + 2 else 2 * numAnchors + i
                        val bhIdx = if (isTransposed) i * (numClasses + 4) + 3 else 3 * numAnchors + i

                        var cx = getValue(cxIdx)
                        var cy = getValue(cyIdx)
                        var bw = getValue(bwIdx)
                        var bh = getValue(bhIdx)

                        // 5. THE FIX: Dynamic Scale Check (Prevents 1-Pixel Boxes)
                        if (cx > 2f || cy > 2f || bw > 2f || bh > 2f) {
                            cx /= yoloInputSize
                            cy /= yoloInputSize
                            bw /= yoloInputSize
                            bh /= yoloInputSize
                        }

                        val x1 = (cx - bw / 2f) * width
                        val y1 = (cy - bh / 2f) * height
                        val x2 = (cx + bw / 2f) * width
                        val y2 = (cy + bh / 2f) * height

                        var avgDepth = 50f

                        // 6. High-Speed Depth Sampling (Center-Weighted + Stride)
                        // ... (Keep your exact Depth Sampling logic here) ...
                        val depthCx = (cx * depthInputSize)
                        val depthCy = (cy * depthInputSize)
                        val depthBoxW = (bw * depthInputSize) * 0.2f
                        val depthBoxH = (bh * depthInputSize) * 0.2f

                        val dx1 = (depthCx - depthBoxW / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dy1 = (depthCy - depthBoxH / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dx2 = (depthCx + depthBoxW / 2f).toInt().coerceIn(0, depthInputSize - 1)
                        val dy2 = (depthCy + depthBoxH / 2f).toInt().coerceIn(0, depthInputSize - 1)

                        var depthSum = 0f
                        var count = 0

                        for (y in dy1..dy2 step 2) {
                            for (x in dx1..dx2 step 2) {
                                depthSum += depthOutputMap[y][x]
                                count++
                            }
                        }
                        if (count > 0) avgDepth = depthSum / count

                        // 7. Pointer Logic
                        val isPointedAt = latestPointerX?.let { px ->
                            latestPointerY?.let { py ->
                                RectF(x1, y1, x2, y2).contains(px, py)
                            }
                        } ?: false

                        rawBoxes.add(DetectedObject(cocoLabels[classId] ?: "object", RectF(x1, y1, x2, y2), avgDepth, isPointedAt))
                    }
                }

                // Apply Non-Maximum Suppression to remove duplicate boxes
                detectedObjects.addAll(applyNMS(rawBoxes, 0.4f))

            } catch (e: Exception) {
                Log.e("VisionPipeline", "YOLO Pipeline failed", e)
            }
        }

        // --- PATHFINDING & ROUTING ---
        val numSectors = 10
        val sectorScores = FloatArray(numSectors)
        var bestSector = -1
        var maxScore = -1f
        var rawNavCmd = "🛑 STOP"

        var aStarPath: List<PointF>? = null

        // FIX 1: Use Float division to perfectly align Depth Sectors with Screen Sectors
        val sectorWDepth = depthInputSize.toFloat() / numSectors
        val sectorWScreen = width / numSectors.toFloat()

        for (i in 0 until numSectors) {
            var depthSum = 0f
            var count = 0

            // Map boundaries precisely using Float offsets cast back to Int
            val startX = (i * sectorWDepth).toInt()
            val endX = ((i + 1) * sectorWDepth).toInt()

            for (y in (depthInputSize * 0.4).toInt() until depthInputSize) {
                for (x in startX until endX) {
                    depthSum += depthOutputMap[y][x]
                    count++
                }
            }

            sectorScores[i] = max(0f, 255f - (if (count > 0) depthSum / count else 0f))

            val sLeft = i * sectorWScreen
            val sRight = (i + 1) * sectorWScreen

            for (obj in detectedObjects) {
                // If obstacle spans into this sector (bottom 60% of screen)
                if (obj.bbox.right > sLeft && obj.bbox.left < sRight && obj.bbox.bottom > height * 0.4f) {
                    // FIX 2: Aggressive Penalty. Lowered threshold (180f = closer) for indoor use.
                    if (obj.distanceMetric > 180f) {
                        sectorScores[i] = min(sectorScores[i], 20f) // Tank the score
                    }
                }
            }

            if (sectorScores[i] > maxScore) {
                maxScore = sectorScores[i]
                bestSector = i
            }
        }

        val maxForwardScore = maxOf(sectorScores[3], sectorScores[4], sectorScores[5], sectorScores[6])
        val forwardBlocked = maxForwardScore < 50f

        if (activeTargets.isNotEmpty()) {
            aStarPath = calculateAStar(depthOutputMap, detectedObjects, activeTargets, width, height)

            if (aStarPath != null && aStarPath.size > 5) {
                val lookAheadIdx = min(aStarPath.size - 1, 5)
                val lookAheadPoint = aStarPath[lookAheadIdx]

                val targetSector = (lookAheadPoint.x / sectorWScreen).toInt().coerceIn(0, numSectors - 1)
                bestSector = targetSector

                rawNavCmd = when {
                    maxScore < 40f -> "🛑 Stop, turn around"
                    forwardBlocked && bestSector in 3..6 -> "🛑 Stop, path blocked"
                    bestSector in 3..6 -> "⬆️ Target Straight"
                    bestSector == 2 -> "↖️ Target Slightly Left"
                    bestSector == 7 -> "↗️ Target Slightly Right"
                    bestSector < 2 -> "⬅️ Target Left"
                    else -> "➡️ Target Right"
                }
            } else {
                rawNavCmd = "🔍 Scanning for target..."
            }
        } else {
            rawNavCmd = when {
                maxScore < 40f -> "🛑 Stop, turn around"
                forwardBlocked -> {
                    if (bestSector < 4) "🛑 Stop, move left"
                    else "🛑 Stop, move right"
                }
                bestSector in 3..6 -> "⬆️ Go Straight"
                bestSector == 2 -> "↖️ Slightly Left"
                bestSector == 7 -> "↗️ Slightly Right"
                bestSector < 2 -> "⬅️ Turn Left"
                else -> "➡️ Turn Right"
            }
        }

        // Add Lateral Object Callouts (only if moving forward/slightly and not already turning/stopping)
        if (!rawNavCmd.contains("Stop") && !rawNavCmd.contains("Scanning")) {
            var leftObj = false
            var rightObj = false
            for (obj in detectedObjects) {
                if (obj.distanceMetric > 170f) { // Close enough to warrant a side warning
                    val objCenterX = (obj.bbox.left + obj.bbox.right) / 2
                    val objSector = (objCenterX / sectorWScreen).toInt().coerceIn(0, 9)
                    if (objSector in 0..1) leftObj = true
                    if (objSector in 8..9) rightObj = true
                }
            }
            if (leftObj && rightObj) rawNavCmd += ". Objects on both sides."
            else if (leftObj) rawNavCmd += ". Object on left."
            else if (rightObj) rawNavCmd += ". Object on right."
        }

        commandHistory.add(rawNavCmd)
        if (commandHistory.size > 3) commandHistory.removeAt(0)
        val smoothedNavCmd = commandHistory.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: rawNavCmd

        val (isCollision, dangerObj) = spatialAnalyzer.checkReflexCollision(detectedObjects)
        onSceneProcessed(detectedObjects.toList(), NavState(smoothedNavCmd, sectorScores, bestSector, width, height, aStarPath), isCollision, dangerObj)
        imageProxy.close()
    }

    private fun calculateAStar(depthMap: Array<FloatArray>, boxes: List<DetectedObject>, targets: List<String>, imgW: Float, imgH: Float): List<PointF>? {
        // 1. Reset gScore and cameFrom rapidly
        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                gScore[y][x] = Float.MAX_VALUE
                cameFrom[y][x] = -1
            }
        }

        val depthH = depthMap.size
        val depthW = depthMap[0].size

        // 2. Populate costMap (In-place)
        for (y in 0 until gridH) {
            for (x in 0 until gridW) {
                // Inside calculateAStar()
                val dy = (y * depthH.toFloat() / gridH).toInt().coerceIn(0, depthH - 1)
                val dx = (x * depthW.toFloat() / gridW).toInt().coerceIn(0, depthW - 1)
                costMap[y][x] = depthMap[dy][dx] * 2f
            }
        }

        // ... (Your existing Target Box logic, writing 0f or 1000f to costMap) ...
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

        // 3. Optimized Queue (Still an object, but stores encoded Ints instead of Pairs)
        // Triple = <F-Score, X, Y>
        val queue = PriorityQueue<Triple<Float, Int, Int>>(compareBy { it.first })
        queue.add(Triple(0f, startX, startY))
        gScore[startY][startX] = 0f

        while (queue.isNotEmpty()) {
            val curr = queue.poll()!!
            val cx = curr.second
            val cy = curr.third

            if ((cx == goalX && cy == goalY) || (abs(cx - goalX) <= 1 && abs(cy - goalY) <= 1)) {
                val path = mutableListOf<PointF>()
                var currX = cx
                var currY = cy

                // Backtrack using the primitive array
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

                    // Diagonal vs Straight movement cost
                    val moveCost = if (d[0] != 0 && d[1] != 0) 14.14f else 10f
                    val tG = gScore[cy][cx] + moveCost + penalty

                    if (tG < gScore[ny][nx]) {
                        cameFrom[ny][nx] = cy * gridW + cx // Encode Y and X
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