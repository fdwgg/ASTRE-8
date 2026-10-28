package io.github.fdwgg.astre8

import android.content.Context
import android.net.Uri
import java.io.File

/** Minimal acoustic-detection integration. Call from a worker thread. */
object Astre8Example {
    fun extendToWave(
        context: Context,
        source: Uri,
        output: File,
        targetMs: Long,
        cancelled: () -> Boolean = { false },
        progress: (String) -> Unit = {}
    ): ExtendedAudio {
        val destination = output.absoluteFile
        require(destination.parentFile?.isDirectory == true) { "Output directory must exist." }
        require(!destination.exists()) { "Choose a new output file." }
        val pcm = File.createTempFile("astre8-", ".pcm", context.cacheDir)
        try {
            val audio = AudioExtender.decode(context, source, pcm, cancelled, progress)
            val region = LoopAnalysis.detect(audio.analysis, cancelled = cancelled)
                ?: error("No suitable loop was detected in this source.")
            return AudioExtender.render(
                audio, region, targetMs, fadeSeconds = 5, crossfadeMs = 30,
                out = destination, cancelled = cancelled, progress = progress
            )
        } finally {
            pcm.delete()
        }
    }
}
