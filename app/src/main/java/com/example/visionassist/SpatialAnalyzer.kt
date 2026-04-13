package com.example.visionassist

import android.graphics.RectF
import kotlin.math.abs

data class DetectedObject(
    val className: String,
    val bbox: RectF,
    val distanceMetric: Float,
    val isPointedAt: Boolean = false
)

class SpatialAnalyzer(
    private val depthTolerance: Float = 20f,
    private val pixelTolerance: Float = 50f
) {
    fun calculateRelations(objects: List<DetectedObject>): List<String> {
        val relations = mutableSetOf<String>()
        val numObjs = objects.size

        for (i in 0 until numObjs) {
            for (j in i + 1 until numObjs) {
                val objA = objects[i]
                val objB = objects[j]

                if (abs(objA.distanceMetric - objB.distanceMetric) > depthTolerance) continue

                val xOverlap = objA.bbox.left < objB.bbox.right && objA.bbox.right > objB.bbox.left
                val yOverlap = objA.bbox.top < objB.bbox.bottom && objA.bbox.bottom > objB.bbox.top

                if (xOverlap && abs(objA.bbox.bottom - objB.bbox.top) < pixelTolerance) {
                    relations.add("${objA.className} is ON ${objB.className}")
                } else if (xOverlap && abs(objB.bbox.bottom - objA.bbox.top) < pixelTolerance) {
                    relations.add("${objB.className} is ON ${objA.className}")
                }

                if (yOverlap && (abs(objA.bbox.right - objB.bbox.left) < pixelTolerance || abs(objB.bbox.right - objA.bbox.left) < pixelTolerance)) {
                    relations.add("${objA.className} is NEXT TO ${objB.className}")
                }
            }
        }
        return relations.toList()
    }

    /** * Hardcoded safety interrupt. Bypasses LLM if object is dangerously close.
     * Width threshold lowered to 80f to ensure narrower obstacles (like chair legs) still trigger a stop.
     */
    fun checkReflexCollision(objects: List<DetectedObject>, dangerDepthThreshold: Float = 200f): Pair<Boolean, String?> {
        for (obj in objects) {
            if (obj.distanceMetric > dangerDepthThreshold && obj.bbox.width() > 80f) {
                return Pair(true, obj.className)
            }
        }
        return Pair(false, null)
    }
}