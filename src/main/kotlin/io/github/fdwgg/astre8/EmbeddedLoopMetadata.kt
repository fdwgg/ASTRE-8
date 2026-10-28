package io.github.fdwgg.astre8

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.util.Locale
import java.util.concurrent.CancellationException

/** Explicit sample coordinates, never a guess based on a title or a repeated short phrase. */
data class EmbeddedLoop(val startFrame: Long, val endFrame: Long, val rate: Int, val format: String)

/**
 * Bounded metadata reader. Does not scan or copy audio data. In particular, a WAV's
 * `smpl` chunk may follow a large `data` chunk and FLAC artwork may precede its tags.
 * A null result means ordinary acoustic detection may continue; cancellation propagates.
 *
 * WAV smpl end coordinates are inclusive (RIFF Multimedia Data Standards Update,
 * April 1994, pp. 17-18). Vorbis LOOPSTART/LOOPEND/LOOPLENGTH are the common
 * sample-count convention, with an exclusive end; time strings are not interpreted.
 * MP3 and Opus tags are deliberately unsupported because encoder delay/pre-skip
 * makes their sample coordinate systems ambiguous without decoder-specific mapping.
 */
object EmbeddedLoopMetadata {
    const val MAX_METADATA_BYTES = 256 * 1024
    private const val MAX_STEPS = 512
    private const val MAX_NANOS = 250_000_000L
    private val LOOP_FIELDS = setOf("LOOPSTART", "LOOPEND", "LOOPLENGTH", "LOOPTYPE", "LOOPMODE")

    /** The caller owns this input. Offsets are relative to this audio file, not its container. */
    fun interface ReadAt {
        fun read(position: Long, target: ByteArray, offset: Int, count: Int): Int
    }

    fun read(file: File, decodedRate: Int, decodedFrames: Long, cancelled: () -> Boolean = { false }): EmbeddedLoop? =
        try {
            RandomAccessFile(file, "r").use { source ->
                read(source.length(), decodedRate, decodedFrames, ReadAt { position, bytes, offset, count ->
                    source.seek(position)
                    source.read(bytes, offset, count)
                }, cancelled)
            }
        } catch (_: IOException) { null }

    fun read(length: Long, decodedRate: Int, decodedFrames: Long, input: ReadAt, cancelled: () -> Boolean = { false }): EmbeddedLoop? {
        if (length < 4 || decodedRate <= 0 || decodedFrames <= 0) return null
        return try {
            val source = Source(length, input, cancelled)
            val signature = source.bytes(0, 4).ascii()
            val loop = when (signature) {
                "RIFF" -> wave(source, decodedFrames)
                "fLaC" -> flac(source, decodedFrames)
                "OggS" -> vorbis(source)
                else -> null
            }
            loop?.takeIf { it.rate == decodedRate && it.startFrame >= 0 && it.endFrame > it.startFrame && it.endFrame <= decodedFrames }
        } catch (_: InvalidMetadata) { null }
          catch (_: IOException) { null }
          catch (_: IllegalArgumentException) { null }
    }

    private class InvalidMetadata : RuntimeException()
    private fun valid(condition: Boolean) { if (!condition) throw InvalidMetadata() }
    private class Source(val length: Long, val input: ReadAt, val cancelled: () -> Boolean) {
        private var readBytes = 0
        private var steps = 0
        private val started = System.nanoTime()
        fun checkpoint() {
            if (cancelled()) throw CancellationException()
            valid(System.nanoTime() - started <= MAX_NANOS)
        }
        fun check() { checkpoint(); valid(++steps <= MAX_STEPS) }
        fun bytes(position: Long, count: Int): ByteArray {
            check()
            valid(count >= 0 && position >= 0 && position <= length && count.toLong() <= length - position)
            valid(count <= MAX_METADATA_BYTES - readBytes)
            val result = ByteArray(count)
            var offset = 0
            while (offset < count) {
                check()
                val read = input.read(position + offset, result, offset, count - offset)
                valid(read > 0 && read <= count - offset)
                offset += read
                readBytes += read
            }
            return result
        }
    }

    private fun wave(source: Source, decodedFrames: Long): EmbeddedLoop? {
        val header = source.bytes(0, 12)
        valid(header.ascii(8, 4) == "WAVE")
        val end = header.le32(4) + 8
        valid(end in 12..source.length)
        var position = 12L
        var rate = 0
        var blockAlign = 0
        var dataFrames: Long? = null
        var loop: Pair<Long, Long>? = null
        while (position < end) {
            source.check()
            valid(end - position >= 8)
            val chunk = source.bytes(position, 8)
            val size = chunk.le32(4)
            val payload = position + 8
            valid(size <= end - payload)
            when (chunk.ascii(0, 4)) {
                "fmt " -> {
                    valid(rate == 0 && size >= 16)
                    val fmt = source.bytes(payload, 16)
                    // Only uncompressed formats have unambiguous source sample coordinates.
                    valid(fmt.le16(0) == 1 || fmt.le16(0) == 3)
                    val channels = fmt.le16(2)
                    rate = fmt.le32(4).toInt()
                    val bits = fmt.le16(14)
                    blockAlign = fmt.le16(12)
                    valid(rate > 0 && channels > 0 && bits in 8..64 && bits % 8 == 0 && blockAlign == channels * (bits / 8))
                }
                "data" -> {
                    valid(dataFrames == null && blockAlign > 0 && size % blockAlign == 0L)
                    dataFrames = size / blockAlign
                }
                "smpl" -> {
                    valid(loop == null && size >= 60)
                    val smpl = source.bytes(payload, 60)
                    // One integer, forward loop. Do not choose between multiple arrangements,
                    // reinterpret alternating/backward loops, or round a fractional boundary.
                    valid(smpl.le32(28) == 1L && smpl.le32(40) == 0L && smpl.le32(52) == 0L)
                    valid(smpl.le32(32) <= size - 60)
                    loop = smpl.le32(44) to smpl.le32(48) + 1
                }
            }
            val paddedSize = size + (size and 1)
            valid(paddedSize <= end - payload)
            position = payload + paddedSize
        }
        valid(dataFrames == decodedFrames && rate > 0)
        return loop?.let { EmbeddedLoop(it.first, it.second, rate, "WAV smpl") }
    }

    private fun flac(source: Source, decodedFrames: Long): EmbeddedLoop? {
        var position = 4L
        var rate = 0
        var streamFrames = 0L
        var tags: Tags? = null
        var first = true
        do {
            val header = source.bytes(position, 4)
            val type = header[0].toInt() and 0x7f
            val last = header[0].toInt() and 0x80 != 0
            val size = ((header[1].toInt() and 255) shl 16) or ((header[2].toInt() and 255) shl 8) or (header[3].toInt() and 255)
            val payload = position + 4
            valid(payload <= source.length && size.toLong() <= source.length - payload && (!first || type == 0))
            when (type) {
                0 -> {
                    valid(first && size == 34)
                    val info = source.bytes(payload, 34)
                    var packed = 0L
                    for (i in 10..17) packed = (packed shl 8) or (info[i].toLong() and 255)
                    rate = (packed ushr 44).toInt()
                    streamFrames = packed and 0xfffffffffL
                }
                4 -> {
                    valid(tags == null && size <= MAX_METADATA_BYTES)
                    tags = comments(source.bytes(payload, size), 0, false, source::checkpoint)
                }
                127 -> throw InvalidMetadata()
            }
            first = false
            position = payload + size
        } while (!last)
        valid(rate > 0 && (streamFrames == 0L || streamFrames == decodedFrames))
        return tags?.loop(rate, "FLAC comments")
    }

    private fun vorbis(source: Source): EmbeddedLoop? {
        var position = 0L
        var serial: Long? = null
        var sequence = 0L
        var rate = 0
        var packets = 0
        var partial = false
        val packet = ByteArrayOutputStream()
        while (packets < 2) {
            val header = source.bytes(position, 27)
            valid(header.ascii(0, 4) == "OggS" && header[4].toInt() == 0)
            val flags = header[5].toInt() and 255
            valid(flags and 0xf8 == 0 && (flags and 1 != 0) == partial)
            valid((sequence == 0L) == (flags and 2 != 0))
            val pageSerial = header.le32(14)
            if (serial == null) serial = pageSerial
            valid(pageSerial == serial && header.le32(18) == sequence++)
            val lacing = source.bytes(position + 27, header[26].toInt() and 255)
            val bodySize = lacing.sumOf { it.toInt() and 255 }
            val body = source.bytes(position + 27 + lacing.size, bodySize)
            valid(oggCrc(header, lacing, body) == header.le32(22))
            var offset = 0
            for (lace in lacing) {
                val size = lace.toInt() and 255
                valid(packet.size() + size <= MAX_METADATA_BYTES)
                packet.write(body, offset, size)
                offset += size
                partial = size == 255
                if (!partial) {
                    val bytes = packet.toByteArray()
                    packet.reset()
                    if (packets++ == 0) {
                        valid(bytes.size == 30 && bytes[0].toInt() == 1 && bytes.ascii(1, 6) == "vorbis")
                        valid(bytes.le32(7) == 0L && bytes[11].toInt() and 255 > 0 && bytes[29].toInt() and 1 != 0)
                        valid(offset == body.size)
                        rate = bytes.le32(12).toInt()
                    } else {
                        valid(bytes.size >= 8 && bytes[0].toInt() == 3 && bytes.ascii(1, 6) == "vorbis")
                        return comments(bytes, 7, true, source::checkpoint).loop(rate, "Vorbis comments")
                    }
                }
            }
            valid(flags and 4 == 0)
            position += 27L + lacing.size + bodySize
        }
        return null
    }

    private fun oggCrc(vararg chunks: ByteArray): Long {
        var crc = 0
        chunks.forEachIndexed { chunkIndex, bytes -> bytes.forEachIndexed { index, byte ->
            val value = if (chunkIndex == 0 && index in 22..25) 0 else byte.toInt() and 255
            crc = crc xor (value shl 24)
            repeat(8) { crc = if (crc < 0) (crc shl 1) xor 0x04c11db7 else crc shl 1 }
        } }
        return crc.toLong() and 0xffffffffL
    }

    private class Tags {
        val values = HashMap<String, Long>()
        fun add(comment: String) {
            val separator = comment.indexOf('=')
            if (separator < 0) return
            val key = comment.substring(0, separator).uppercase(Locale.ROOT).replace("_", "")
            if (key !in LOOP_FIELDS) return
            val value = comment.substring(separator + 1).trim()
            if (key == "LOOPTYPE" || key == "LOOPMODE") {
                valid(value == "0" || value.equals("forward", true))
                return
            }
            valid(value.isNotEmpty() && value.length <= 19 && value.all { it in '0'..'9' })
            val number = value.toLongOrNull() ?: throw InvalidMetadata()
            val previous = values.put(key, number)
            valid(previous == null || previous == number)
        }
        fun loop(rate: Int, format: String): EmbeddedLoop? {
            val start = values["LOOPSTART"] ?: return null
            val end = values["LOOPEND"]
            val length = values["LOOPLENGTH"]
            if (end == null && length == null) return null
            if (length != null) valid(length > 0 && length <= Long.MAX_VALUE - start && (end == null || end == start + length))
            return EmbeddedLoop(start, end ?: (start + length!!), rate, format)
        }
    }

    private fun comments(bytes: ByteArray, start: Int, framing: Boolean, checkpoint: () -> Unit): Tags {
        var position = start
        fun count(): Int {
            valid(position <= bytes.size - 4)
            val count = bytes.le32(position)
            position += 4
            valid(count <= bytes.size - position)
            return count.toInt()
        }
        val vendorLength = count()
        position += vendorLength
        val fields = count()
        valid(fields <= 4096 && fields <= (bytes.size - position) / 4)
        val tags = Tags()
        repeat(fields) {
            if (it % 32 == 0) checkpoint()
            val size = count()
            val decoder = Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).onUnmappableCharacter(CodingErrorAction.REPORT)
            val field = decoder.decode(ByteBuffer.wrap(bytes, position, size)).toString()
            tags.add(field)
            position += size
        }
        valid(if (framing) position == bytes.size - 1 && bytes[position].toInt() and 1 != 0 else position == bytes.size)
        return tags
    }

    private fun ByteArray.ascii(offset: Int = 0, count: Int = size): String = String(this, offset, count, Charsets.US_ASCII)
    private fun ByteArray.le16(offset: Int): Int = (this[offset].toInt() and 255) or ((this[offset + 1].toInt() and 255) shl 8)
    private fun ByteArray.le32(offset: Int): Long = le16(offset).toLong() or (le16(offset + 2).toLong() shl 16)
}
