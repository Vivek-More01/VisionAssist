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
import android.util.Log
import java.util.Locale

class VoiceAgent(
    private val context: Context,
    private val onCommandCaptured: (String) -> Unit
) : TextToSpeech.OnInitListener {

    private var tts: TextToSpeech? = null
    private var speechRecognizer: SpeechRecognizer? = null
    private var isTtsReady = false
    private val mainHandler = Handler(Looper.getMainLooper()) // Enforces UI thread

    private val recognitionIntent: Intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        // FIX 1: Force Offline execution for zero-latency edge processing
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
    }

    init {
        tts = TextToSpeech(context, this)
        // Ensure creation happens on the Main Thread
        mainHandler.post { setupSpeechRecognizer() }
    }

    private fun setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) { Log.d("VoiceAgent", "Ready for speech") }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}

            override fun onError(error: Int) {
                Log.e("VoiceAgent", "STT Error Code: $error")
                // Optional: Provide audio feedback if the mic timed out without hearing anything
                if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                    speak("I didn't catch that.")
                }
            }

            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                if (!matches.isNullOrEmpty()) {
                    onCommandCaptured(matches[0])
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val result = tts?.setLanguage(Locale.US)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                Log.e("VoiceAgent", "TTS Language not supported")
            } else {
                tts?.setPitch(1.0f)
                tts?.setSpeechRate(1.1f) // Slightly faster for real-time nav
                isTtsReady = true // FIX 2: Mark as ready
            }
        } else {
            Log.e("VoiceAgent", "TTS Initialization failed")
        }
    }

    fun speak(text: String, force: Boolean = false) {
        if (!isTtsReady) {
            Log.w("VoiceAgent", "TTS not ready yet, dropping text: $text")
            return
        }

        val queueMode = if (force) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD
        val utteranceId = "NavUtterance_${System.currentTimeMillis()}"
        tts?.speak(text, queueMode, null, utteranceId)
    }

    fun listenForCommand() {
        // FIX 3: Push to Main Thread to prevent crashes if called from coroutines
        mainHandler.post {
            tts?.stop()
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