package com.example.visionassist

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

enum class SpeechPriority {
    CRITICAL,
    HIGH,
    NORMAL
}

class VoiceAgent(
    private val context: Context,
    private val onCommandCaptured: (String) -> Unit
) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var speechRecognizer: SpeechRecognizer? = null
    private var isTtsReady = false
    private val mainHandler = Handler(Looper.getMainLooper())

    // State Tracking
    @Volatile private var currentPriority: SpeechPriority? = null
    @Volatile private var isSpeaking = false

    @Volatile private var lastCriticalCommand = ""
    @Volatile private var lastCriticalSpeakTime = 0L

    @Volatile private var lastHighCommand = ""
    @Volatile private var lastHighSpeakTime = 0L

    @Volatile private var lastNormalSpeakTime = 0L

    private val recognitionIntent: Intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
    }

    init {
        tts = TextToSpeech(context, this)
        // We defer setup to when the user actually taps the screen to prevent dead-listeners
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val availableVoices = tts?.voices ?: emptySet()
            val selectedVoice = availableVoices.find { voice ->
                voice.locale.language == "en" &&
                        voice.locale.country == "IN" &&
                        voice.name.contains("network", ignoreCase = true)
            } ?: availableVoices.find { it.locale.language == "en" && it.locale.country == "IN" }

            if (selectedVoice != null) {
                tts?.voice = selectedVoice
            } else {
                tts?.language = Locale("en", "IN")
            }

            tts?.setPitch(1.0f)
            tts?.setSpeechRate(1.2f)

            // THE FIX: Listen to ALL possible termination states to prevent Ghost Locks
            tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    isSpeaking = true
                }
                override fun onDone(utteranceId: String?) {
                    isSpeaking = false
                    currentPriority = null
                }
                @Deprecated("Deprecated in Java")
                override fun onError(utteranceId: String?) {
                    isSpeaking = false
                    currentPriority = null
                }
                // Catches QUEUE_FLUSH interruptions
                override fun onStop(utteranceId: String?, interrupted: Boolean) {
                    isSpeaking = false
                    currentPriority = null
                }
            })
            isTtsReady = true
        }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                // Ignore silent timeouts, but notify if speech was completely missed
                if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    speak("I didn't catch that.", SpeechPriority.HIGH)
                }
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) onCommandCaptured(matches[0])
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    fun speak(text: String, priority: SpeechPriority = SpeechPriority.NORMAL) {
        if (!isTtsReady || text.isBlank()) return
        val currentTime = System.currentTimeMillis()

        when (priority) {
            SpeechPriority.CRITICAL -> {
                if (text == lastCriticalCommand && isSpeaking && (currentTime - lastCriticalSpeakTime < 2500L)) return

                currentPriority = SpeechPriority.CRITICAL
                isSpeaking = true
                lastCriticalCommand = text
                lastCriticalSpeakTime = currentTime

                // THE FIX: If TTS rejects the string, instantly unlock
                val status = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "CRITICAL_$currentTime")
                if (status == TextToSpeech.ERROR) isSpeaking = false

                mainHandler.postDelayed({ lastCriticalCommand = "" }, 2500L)
            }

            SpeechPriority.HIGH -> {
                if (currentPriority == SpeechPriority.CRITICAL && isSpeaking) return
                if (text == lastHighCommand && (currentTime - lastHighSpeakTime < 4000L)) return

                currentPriority = SpeechPriority.HIGH
                isSpeaking = true
                lastHighCommand = text
                lastHighSpeakTime = currentTime

                val status = tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "HIGH_$currentTime")
                if (status == TextToSpeech.ERROR) isSpeaking = false

                mainHandler.postDelayed({ lastHighCommand = "" }, 4000L)
            }

            SpeechPriority.NORMAL -> {
                if (isSpeaking) return
                if (currentTime - lastNormalSpeakTime < 5000L) return

                currentPriority = SpeechPriority.NORMAL
                isSpeaking = true
                lastNormalSpeakTime = currentTime

                val status = tts?.speak(text, TextToSpeech.QUEUE_ADD, null, "NORMAL_$currentTime")
                if (status == TextToSpeech.ERROR) isSpeaking = false
            }
        }
    }

    // THE FIX: Self-Healing Listener
    fun listenForCommand() {
        mainHandler.post {
            // Force reset TTS state
            tts?.stop()
            isSpeaking = false
            currentPriority = null

            // Destroy the old instance and build a fresh one to guarantee it listens
            speechRecognizer?.destroy()
            setupSpeechRecognizer()
            speechRecognizer?.startListening(recognitionIntent)
        }
    }

    fun shutdown() {
        mainHandler.post {
            tts?.stop()
            tts?.shutdown()
            speechRecognizer?.destroy()
        }
    }
}