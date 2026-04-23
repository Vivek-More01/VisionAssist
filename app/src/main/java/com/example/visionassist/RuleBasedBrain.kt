package com.example.visionassist

import android.util.Log

data class AgentAction(
    val intent: String,
    val targetObjects: List<String>,
    val action: String,
    val voiceResponse: String
)

class RuleBasedBrain {

    // 1. Entity Resolution Map: Maps spoken synonyms to exact COCO labels
    private val entityMap = mapOf(
        "person" to "person", "man" to "person", "woman" to "person", "human" to "person",
        "chair" to "chair", "seat" to "chair",
        "couch" to "couch", "sofa" to "couch",
        "potted plant" to "potted plant", "plant" to "potted plant",
        "bed" to "bed",
        "dining table" to "dining table", "table" to "dining table", "desk" to "dining table",
        "tv" to "tv", "television" to "tv", "screen" to "tv", "monitor" to "tv",
        "keyboard" to "keyboard",
        "cell phone" to "cell phone", "phone" to "cell phone", "mobile" to "cell phone",
        "book" to "book"
    )

    fun processCommand(voiceCommand: String): AgentAction {
        val lowerCmd = voiceCommand.lowercase().trim()
        Log.i("RuleBasedBrain", "Processing Voice Command: $lowerCmd")

        // 1. DESCRIBE COMMAND
        if (lowerCmd.contains("describe") || lowerCmd.contains("what is around") || lowerCmd.contains("what do you see")) {
            return AgentAction(
                intent = "describe",
                targetObjects = emptyList(),
                action = "describe_scene",
                voiceResponse = "Analyzing scene..."
            )
        }

        // 2. ROBUST SEEK COMMAND (Entity Matching)
        val seekPrefixes = listOf("find", "where is", "look for", "search for")
        if (seekPrefixes.any { lowerCmd.contains(it) }) {
            // Search the command for any known entity
            for ((synonym, cocoLabel) in entityMap) {
                // Pad with word boundaries to prevent "plant" matching inside "plantation"
                if (lowerCmd.matches(Regex(".*\\b$synonym\\b.*"))) {
                    return AgentAction(
                        intent = "seek",
                        targetObjects = listOf(cocoLabel),
                        action = "scan",
                        voiceResponse = "Looking for $cocoLabel. Please pan the camera."
                    )
                }
            }

            // If they asked to find something not in the dictionary
            return AgentAction(
                intent = "unknown",
                targetObjects = emptyList(),
                action = "scan",
                voiceResponse = "I am not trained to detect that object yet."
            )
        }

        // 3. EXPLORE/DEFAULT COMMAND
        if (lowerCmd.contains("navigate") || lowerCmd.contains("go") || lowerCmd.contains("start") || lowerCmd.contains("resume")) {
            return AgentAction(
                intent = "explore",
                targetObjects = emptyList(),
                action = "go_straight",
                voiceResponse = "Resuming obstacle avoidance."
            )
        }

        // 4. STOP COMMAND
        if (lowerCmd.contains("stop") || lowerCmd.contains("halt") || lowerCmd.contains("pause")) {
            return AgentAction(
                intent = "stop",
                targetObjects = emptyList(),
                action = "stop",
                voiceResponse = "Navigation paused."
            )
        }

        // 5. FALLBACK
        return AgentAction(
            intent = "unknown",
            targetObjects = emptyList(),
            action = "scan",
            voiceResponse = "Command not recognized. Say 'describe scene', 'find an object', or 'navigate'."
        )
    }
}