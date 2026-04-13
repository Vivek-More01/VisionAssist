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

        // 1. SEEK COMMAND ("find chair", "where is the door")
        if (lowerCmd.contains("find") || lowerCmd.contains("where is") || lowerCmd.contains("look for")) {
            val target = extractTarget(lowerCmd)
            if (target.isNotEmpty()) {
                return AgentAction(
                    intent = "seek",
                    targetObjects = listOf(target),
                    action = "scan",
                    voiceResponse = "Looking for $target."
                )
            }
        }

        // 2. EXPLORE COMMAND ("navigate", "go")
        if (lowerCmd.contains("navigate") || lowerCmd.contains("go") || lowerCmd.contains("start")) {
            return AgentAction(
                intent = "explore",
                targetObjects = emptyList(), // Clears targets, enters general avoidance mode
                action = "go_straight",
                voiceResponse = "Starting navigation."
            )
        }

        // 3. STOP COMMAND
        if (lowerCmd.contains("stop") || lowerCmd.contains("halt")) {
            return AgentAction(
                intent = "stop",
                targetObjects = emptyList(),
                action = "stop",
                voiceResponse = "Navigation stopped."
            )
        }

        // 4. FALLBACK
        return AgentAction(
            intent = "unknown",
            targetObjects = emptyList(),
            action = "scan",
            voiceResponse = "Command not recognized. Please say find an object, or navigate."
        )
    }

    private fun extractTarget(command: String): String {
        // Simple regex-style extraction
        val prefixes = listOf("find the ", "find a ", "find ", "where is the ", "where is a ", "where is ", "look for the ", "look for a ", "look for ")
        for (prefix in prefixes) {
            if (command.contains(prefix)) {
                // Returns the immediate next word (e.g., "chair", "person")
                return command.substringAfter(prefix).trim().split(Regex("\\s+")).firstOrNull() ?: ""
            }
        }
        return ""
    }
}