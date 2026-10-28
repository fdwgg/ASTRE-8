package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

class EmbeddedLoopMetadataTest {
    @get:Rule val temp = TemporaryFolder()
    private fun read(bytes: ByteArray, rate: Int = 1000, frames: Long = 1000): EmbeddedLoop? =
        EmbeddedLoopMetadata.read(bytes.size.toLong(), rate, frames, EmbeddedLoopMetadata.ReadAt { position, out, offset, count ->
            if (position >= bytes.size) -1 else minOf(count, bytes.size - position.toInt()).also { bytes.copyInto(out, offset, position.toInt(), position.toInt() + it) }
        })
    private fun ints(vararg values: Long): ByteArray = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { values.forEach { putInt(it.toInt()) } }.array()
    private fun text(value: String) = value.toByteArray(Charsets.UTF_8)
    private fun chunk(name: String, body: ByteArray) = text(name) + ints(body.size.toLong()) + body + ByteArray(body.size % 2)
    private fun fmt(rate: Int = 1000, type: Int = 1): ByteArray = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN).putShort(type.toShort()).putShort(1).putInt(rate).putInt(rate * 2).putShort(2).putShort(16).array()
    private fun smpl(start: Long = 100, inclusiveEnd: Long = 799, type: Long = 0, fraction: Long = 0, loops: Long = 1) =
        ints(0, 0, 1_000_000, 60, 0, 0, 0, loops, 0, 0, type, start, inclusiveEnd, fraction, 0)
    private fun wav(smpl: ByteArray = smpl(), format: ByteArray = fmt(), extra: ByteArray = ByteArray(0)): ByteArray {
        val body = text("WAVE") + chunk("fmt ", format) + chunk("JUNK", byteArrayOf(1)) + chunk("data", ByteArray(2000)) + chunk("smpl", smpl) + extra
        return text("RIFF") + ints(body.size.toLong()) + body
    }
    private fun comments(vararg values: String, vendor: String = "Muzbox test"): ByteArray {
        val vendorBytes = text(vendor)
        return ints(vendorBytes.size.toLong()) + vendorBytes + ints(values.size.toLong()) + values.fold(ByteArray(0)) { all, value -> val bytes = text(value); all + ints(bytes.size.toLong()) + bytes }
    }
    private fun flacBlock(type: Int, last: Boolean, bytes: ByteArray): ByteArray = byteArrayOf((type or if (last) 128 else 0).toByte(), (bytes.size ushr 16).toByte(), (bytes.size ushr 8).toByte(), bytes.size.toByte()) + bytes
    private fun streamInfo(rate: Int = 1000, frames: Long = 1000): ByteArray = ByteArray(34).also { bytes ->
        var packed = (rate.toLong() shl 44) or (15L shl 36) or frames
        for (i in 17 downTo 10) { bytes[i] = packed.toByte(); packed = packed ushr 8 }
    }
    private fun flac(vararg values: String, rate: Int = 1000, frames: Long = 1000): ByteArray = text("fLaC") + flacBlock(0, false, streamInfo(rate, frames)) + flacBlock(4, true, comments(*values))

    @Test fun wavReadsInclusiveForwardLoopAfterPaddedChunks() {
        val file = temp.newFile("tagged.wav").apply { writeBytes(wav()) }
        assertEquals(EmbeddedLoop(100, 800, 1000, "WAV smpl"), EmbeddedLoopMetadata.read(file, 1000, 1000))
    }

    @Test fun wavRejectsUnsupportedOrAmbiguousArrangements() {
        for (metadata in listOf(smpl(type = 1), smpl(type = 2), smpl(fraction = 1), smpl(loops = 2), smpl(start = 900), smpl(inclusiveEnd = 1000))) assertNull(read(wav(metadata)))
        assertNull(read(wav(extra = chunk("smpl", smpl()))))
        assertNull(read(wav(format = fmt(type = 17))))
        assertNull(read(wav(), rate = 2000))
        assertNull(read(wav(), frames = 999))
    }

    @Test fun wavSeeksPastLargeAudioPayloadWithoutReadingIt() {
        val dataBytes = 500_000_000L
        val tail = chunk("smpl", smpl(inclusiveEnd = 999))
        val length = 44L + dataBytes + tail.size
        val header = text("RIFF") + ints(length - 8) + text("WAVE") + chunk("fmt ", fmt(48000)) + text("data") + ints(dataBytes)
        var bytesRead = 0
        val input = EmbeddedLoopMetadata.ReadAt { position, out, offset, count ->
            val block: ByteArray
            val index: Int
            if (position < header.size) { block = header; index = position.toInt() }
            else { assertTrue("Audio data must be skipped", position >= 44L + dataBytes); block = tail; index = (position - 44L - dataBytes).toInt() }
            minOf(count, block.size - index).also { block.copyInto(out, offset, index, index + it); bytesRead += it }
        }
        assertEquals(EmbeddedLoop(100, 1000, 48000, "WAV smpl"), EmbeddedLoopMetadata.read(length, 48000, dataBytes / 2, input))
        assertTrue(bytesRead < 200)
    }

    @Test fun flacSupportsExplicitExclusiveEndOrLengthAndMatchingDuplicates() {
        assertEquals(EmbeddedLoop(100, 800, 1000, "FLAC comments"), read(flac("LOOPSTART=100", "LOOPLENGTH=700")))
        assertEquals(EmbeddedLoop(100, 800, 1000, "FLAC comments"), read(flac("loop_start=100", "LoopEnd=800", "LOOP_LENGTH=700", "LOOPSTART=100", "LOOPMODE=forward")))
    }

    @Test fun flacDoesNotGuessMissingBoundariesOrAcceptConflictsAndTimeStrings() {
        for (tags in listOf(
            arrayOf("LOOPSTART=100"), arrayOf("LOOPEND=800"), arrayOf("LOOPLENGTH=800"),
            arrayOf("LOOPSTART=100", "LOOPEND=800", "LOOPLENGTH=699"),
            arrayOf("LOOPSTART=100", "LOOPSTART=101", "LOOPEND=800"),
            arrayOf("LOOPSTART=0:01", "LOOPEND=800"), arrayOf("LOOPSTART=1.5", "LOOPEND=800"),
            arrayOf("LOOPSTART=-1", "LOOPEND=800"), arrayOf("LOOPSTART=1e2", "LOOPEND=800"),
            arrayOf("LOOPSTART=100", "LOOPLENGTH=9223372036854775807"),
            arrayOf("LOOPSTART=100", "LOOPEND=1001"), arrayOf("LOOPSTART=100", "LOOPEND=100"),
            arrayOf("LOOPSTART=100", "LOOPEND=800", "LOOPTYPE=1"),
            arrayOf("LOOPSTART=100", "LOOPEND=800", "LOOPMODE=pingpong")
        )) assertNull(tags.contentToString(), read(flac(*tags)))
        assertNull(read(flac("LOOPSTART=100", "LOOPEND=800", frames = 999)))
        assertNull(read(flac("LOOPSTART=100", "LOOPEND=800", rate = 2000)))
    }

    @Test fun flacSkipsArtworkAndRejectsOversizedCommentBeforeAllocating() {
        val info = text("fLaC") + flacBlock(0, false, streamInfo())
        val pictureSize = 8_000_000
        val pictureHeader = byteArrayOf(6, (pictureSize ushr 16).toByte(), (pictureSize ushr 8).toByte(), pictureSize.toByte())
        val prefix = info + pictureHeader
        val tags = flacBlock(4, true, comments("LOOPSTART=100", "LOOPEND=800"))
        val tagOffset = prefix.size.toLong() + pictureSize
        var bytesRead = 0
        assertNotNull(EmbeddedLoopMetadata.read(tagOffset + tags.size, 1000, 1000, EmbeddedLoopMetadata.ReadAt { position, out, offset, count ->
            val block: ByteArray
            val index: Int
            if (position < prefix.size) { block = prefix; index = position.toInt() }
            else { assertTrue(position >= tagOffset); block = tags; index = (position - tagOffset).toInt() }
            minOf(count, block.size - index).also { block.copyInto(out, offset, index, index + it); bytesRead += it }
        }))
        assertTrue(bytesRead < 300)
        val hugeSize = EmbeddedLoopMetadata.MAX_METADATA_BYTES + 1
        val hugeHeader = byteArrayOf(0x84.toByte(), (hugeSize ushr 16).toByte(), (hugeSize ushr 8).toByte(), hugeSize.toByte())
        val hugePrefix = info + hugeHeader
        assertNull(EmbeddedLoopMetadata.read(hugePrefix.size.toLong() + hugeSize, 1000, 1000, EmbeddedLoopMetadata.ReadAt { position, out, offset, count ->
            assertTrue(position + count <= hugePrefix.size)
            hugePrefix.copyInto(out, offset, position.toInt(), position.toInt() + count); count
        }))
    }

    private fun oggPage(body: ByteArray, lacing: ByteArray, sequence: Int, flags: Int, serial: Int = 77): ByteArray {
        val header = ByteBuffer.allocate(27).order(ByteOrder.LITTLE_ENDIAN).put(text("OggS")).put(0).put(flags.toByte()).putLong(0).putInt(serial).putInt(sequence).putInt(0).put(lacing.size.toByte()).array()
        val result = header + lacing + body
        var crc = 0
        result.forEach { byte -> crc = crc xor ((byte.toInt() and 255) shl 24); repeat(8) { crc = if (crc < 0) (crc shl 1) xor 0x04c11db7 else crc shl 1 } }
        ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN).putInt(22, crc)
        return result
    }
    private fun identification() = byteArrayOf(1) + text("vorbis") + ints(0) + byteArrayOf(1) + ints(1000, 0, 0, 0) + byteArrayOf(0x66, 1)
    private fun ogg(split: Boolean = false, secondFlags: Int = if (split) 1 else 0, secondSequence: Int = if (split) 2 else 1): ByteArray {
        val id = identification()
        val first = oggPage(id, byteArrayOf(id.size.toByte()), 0, 2)
        val comment = byteArrayOf(3) + text("vorbis") + comments("LOOPSTART=100", "LOOPLENGTH=700", vendor = if (split) "V".repeat(300) else "Muzbox test") + byteArrayOf(1)
        return if (split) first + oggPage(comment.copyOfRange(0, 255), byteArrayOf(255.toByte()), 1, 0) + oggPage(comment.copyOfRange(255, comment.size), byteArrayOf((comment.size - 255).toByte()), secondSequence, secondFlags)
        else first + oggPage(comment, byteArrayOf(comment.size.toByte()), secondSequence, secondFlags)
    }

    @Test fun vorbisReadsCrcValidatedCommentsIncludingContinuedPacket() {
        val expected = EmbeddedLoop(100, 800, 1000, "Vorbis comments")
        assertEquals(expected, read(ogg()))
        assertEquals(expected, read(ogg(split = true)))
    }

    @Test fun vorbisRejectsCorruptionMissingContinuationAndWrongPageSequence() {
        assertNull(read(ogg().also { it[it.lastIndex - 2] = (it[it.lastIndex - 2].toInt() xor 1).toByte() }))
        assertNull(read(ogg(split = true, secondFlags = 0)))
        assertNull(read(ogg(split = true, secondSequence = 3)))
        assertNull(read(ogg(secondFlags = 1)))
        assertNull(read(ogg(secondSequence = 2)))
    }

    @Test fun truncatedAndMalformedMetadataSafelyReturnsNull() {
        for (source in listOf(wav(), flac("LOOPSTART=100", "LOOPEND=800"), ogg(split = true))) {
            for (length in 0 until source.size) assertNull("length=$length", read(source.copyOf(length)))
        }
        assertNull(read(text("ID3") + ByteArray(200)))
        assertNull(read(text("OggS") + ByteArray(40)))
        assertNull(read(text("fLaC") + byteArrayOf(0x80.toByte(), 0, 0, 34)))
    }

    @Test fun cancellationPropagatesAndTinyPartialReadsAreSupported() {
        val bytes = flac("LOOPSTART=100", "LOOPEND=800")
        var calls = 0
        val input = EmbeddedLoopMetadata.ReadAt { position, out, offset, count ->
            calls++; val size = minOf(count, 3, bytes.size - position.toInt()); bytes.copyInto(out, offset, position.toInt(), position.toInt() + size); size
        }
        assertNotNull(EmbeddedLoopMetadata.read(bytes.size.toLong(), 1000, 1000, input))
        calls = 0
        try {
            EmbeddedLoopMetadata.read(bytes.size.toLong(), 1000, 1000, input) { calls >= 3 }
            fail("Cancellation must not be turned into a missing metadata result")
        } catch (_: CancellationException) { }
    }
}
