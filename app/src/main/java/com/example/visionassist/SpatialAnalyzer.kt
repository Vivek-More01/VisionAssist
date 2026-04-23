package com.example.visionassist

import android.graphics.RectF
import kotlin.math.abs

data class DetectedObject(
    val className: String,
    val bbox: RectF,
    val distanceMetric: Float,
    val isPointedAt: Boolean = false
)

class SpatialAnalyzer {

    fun checkReflexCollision(objects: List<DetectedObject>, dangerDepthThreshold: Float = 225f): Pair<Boolean, String?> {
        for (obj in objects) {
            if (obj.distanceMetric > dangerDepthThreshold) {
                return Pair(true, obj.className)
            }
        }
        return Pair(false, null)
    }

    fun calculateRelations(objects: List<DetectedObject>): List<String> {
        val relations = mutableListOf<String>()
        val n = objects.size

        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val obj1 = objects[i]
                val obj2 = objects[j]

                if (obj1.className == obj2.className) continue

                val b1 = obj1.bbox
                val b2 = obj2.bbox

                val cx1 = b1.centerX()
                val cy1 = b1.centerY()
                val cx2 = b2.centerX()
                val cy2 = b2.centerY()

                // NEW: Bifurcated Depth Thresholds
                val depthDiff = abs(obj1.distanceMetric - obj2.distanceMetric)
                val strictPlane = depthDiff < 30f   // For lateral proximity (Next To)
                val relaxedPlane = depthDiff < 80f  // Breathing room for deep surfaces (On/Under)

// Relationship 1: "ON" vs "BEHIND"
                val isHorizontallyAligned = (cx1 > b2.left && cx1 < b2.right) || (cx2 > b1.left && cx2 < b1.right)

                if (isHorizontallyAligned) {
                    if (relaxedPlane) { // Use relaxed breathing room for surfaces
                        if (b1.bottom <= b2.centerY() && b1.bottom >= b2.top - 150f) {
                            relations.add("${obj1.className} is on the ${obj2.className}")
                            continue
                        } else if (b2.bottom <= b1.centerY() && b2.bottom >= b1.top - 150f) {
                            relations.add("${obj2.className} is on the ${obj1.className}")
                            continue
                        }
                    } else {
                        // Only trigger "Behind" if they completely fail the relaxed plane check
                        if (obj1.distanceMetric < obj2.distanceMetric) {
                            relations.add("${obj1.className} is behind the ${obj2.className}")
                            continue
                        } else {
                            relations.add("${obj2.className} is behind the ${obj1.className}")
                            continue
                        }
                    }
                }

// Relationship 2: "NEXT TO"
                val isVerticallyAligned = (cy1 > b2.top && cy1 < b2.bottom) || (cy2 > b1.top && cy2 < b1.bottom)

                if (isVerticallyAligned && strictPlane) { // Strict constraint to prevent false lateral matches
                    val dist1 = abs(b1.right - b2.left)
                    val dist2 = abs(b2.right - b1.left)

                    if (dist1 < 250f || dist2 < 250f) {
                        if (cx1 < cx2) {
                            relations.add("${obj1.className} is to the left of the ${obj2.className}")
                        } else {
                            relations.add("${obj1.className} is to the right of the ${obj2.className}")
                        }
                    }
                }
            }
        }

        return relations.distinct().take(3)
    }
}