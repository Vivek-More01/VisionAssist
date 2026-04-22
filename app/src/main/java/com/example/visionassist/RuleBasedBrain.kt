package com.example.visionassist

import android.util.Log

data class AgentAction(
    val intent: String,
    val targetObjects: List<String>,
    val action: String,
    val voiceResponse: String
)

class RuleBasedBrain {

    fun processCommand(voiceCommand: String): AgentAction {
        val lowerCmd = voiceCommand.lowercase().trim()
        Log.i("RuleBasedBrain", "Processing Voice Command: $lowerCmd")

        // 1. ROBUST SEEK COMMAND (Extracts the exact object name)
        val seekPrefixes = listOf("find the ", "find a ", "find ", "where is the ", "where is a ", "where is ", "look for the ", "look for a ", "look for ")
        for (prefix in seekPrefixes) {
            if (lowerCmd.contains(prefix)) {
                // Splits by space/punctuation and grabs the very next word
                val target = lowerCmd.substringAfter(prefix).trim().split(Regex("[\\s.,!?]+")).firstOrNull() ?: ""
                if (target.isNotEmpty() && target.length > 1) {
                    return AgentAction(
                        intent = "seek",
                        targetObjects = listOf(target),
                        action = "scan",
                        voiceResponse = "Looking for $target. Please pan the camera."
                    )
                }
            }
        }

        // 2. EXPLORE/DEFAULT COMMAND
        if (lowerCmd.contains("navigate") || lowerCmd.contains("go") || lowerCmd.contains("start") || lowerCmd.contains("resume")) {
            return AgentAction(
                intent = "explore",
                targetObjects = emptyList(), // Clears targets to resume general avoidance
                action = "go_straight",
                voiceResponse = "Resuming obstacle avoidance."
            )
        }

        // 3. STOP COMMAND
        if (lowerCmd.contains("stop") || lowerCmd.contains("halt") || lowerCmd.contains("pause")) {
            return AgentAction(
                intent = "stop",
                targetObjects = emptyList(),
                action = "stop",
                voiceResponse = "Navigation paused."
            )
        }

        // 4. FALLBACK
        return AgentAction(
            intent = "unknown",
            targetObjects = emptyList(),
            action = "scan",
            voiceResponse = "Command not recognized. Say 'find' an object, or 'navigate'."
        )
    }
}