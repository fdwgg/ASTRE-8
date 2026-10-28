package io.github.fdwgg.astre8

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Test-only JVM oracle. Compile with the four original engine files. */
fun main(args: Array<String>) {
    when (args[0]) {
        "detect" -> {
            val buffer = ByteBuffer.wrap(File(args[1]).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val pcm = FloatArray(buffer.remaining()).also { buffer.get(it) }
            val loop = LoopAnalysis.detect(pcm, strict = args.getOrNull(2) != "false")
            if (loop == null) println("null") else println(
                "{\"start\":${loop.start},\"end\":${loop.end},\"kind\":\"${loop.kind}\",\"confidence\":${loop.confidence},\"fading\":${loop.fading}}"
            )
        }
        "render" -> {
            val plan = ExtensionPlan(args[3].toInt(), args[4].toInt(), args[5].toLong(), args[6].toLong(),
                args[7].toInt(), 1f, 1f, LoopKind.MANUAL, args[8].toLong())
            ExtensionWaveReader(File(args[1]), plan, args[9].toLong(), args[10].toInt(), args[11].toLong()).use { reader ->
                File(args[2]).outputStream().use { output ->
                    val buffer = ByteArray(65537)
                    var position = 0L
                    while (position < reader.length) {
                        val count = reader.read(position, buffer, 0, buffer.size)
                        check(count > 0)
                        output.write(buffer, 0, count)
                        position += count
                    }
                }
            }
        }
        else -> error("Unknown reference action")
    }
}
