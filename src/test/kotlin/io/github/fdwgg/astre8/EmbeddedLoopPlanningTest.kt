package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sin
import kotlin.math.roundToInt

class EmbeddedLoopPlanningTest {
    @get:Rule val temp = TemporaryFolder()
    private fun source(start: Long = 1234, end: Long = 38321, channels: Int = 1, metadataRate: Int = 44100): DecodedAudio {
        val rate = 44100
        val frames = 44100L
        val pcm = ByteBuffer.allocate((frames * channels * 2).toInt()).order(ByteOrder.LITTLE_ENDIAN)
        repeat(frames.toInt()) { frame -> repeat(channels) { channel -> pcm.putShort((sin(frame * 0.071 + channel) * 21000).roundToInt().toShort()) } }
        return DecodedAudio(temp.newFile().apply { writeBytes(pcm.array()) }, rate, channels, frames, FloatArray(2000), EmbeddedLoop(start, end, metadataRate, "test sample tags"))
    }
    private fun region(audio: DecodedAudio, kind: LoopKind = LoopKind.EMBEDDED): LoopRegion {
        val loop = requireNotNull(audio.embeddedLoop)
        return LoopRegion(loop.startFrame * LoopAnalysis.RATE / audio.rate, loop.endFrame * LoopAnalysis.RATE / audio.rate, 0.3f, 1f, false, kind)
    }

    @Test fun embeddedCoordinatesAvoidAnalysisGridQuantizationAndSeamRefinement() {
        val audio = source()
        val region = region(audio)
        assertNotEquals(audio.embeddedLoop!!.startFrame, region.start * audio.rate / LoopAnalysis.RATE)
        assertNotEquals(audio.embeddedLoop.endFrame, region.end * audio.rate / LoopAnalysis.RATE)
        val plan = AudioExtender.plan(audio, region, 20) { fail("Authored coordinates must not enter acoustic seam refinement"); false }
        assertEquals(1234L, plan.start)
        assertEquals(38321L, plan.end)
        assertEquals(LoopKind.EMBEDDED, plan.kind)
        assertEquals(37087L, ExtensionTimeline(plan.start, plan.end, plan.overlap).period)
    }

    @Test fun zeroStartAuthoredLoopKeepsExactPeriodDespiteRequestedCrossfade() {
        val audio = source(start = 0, end = 38321)
        val plan = AudioExtender.plan(audio, region(audio), 500) { false }
        assertEquals(0, plan.overlap)
        val timeline = ExtensionTimeline(plan.start, plan.end, plan.overlap)
        assertEquals(38321L, timeline.period)
        for (pass in 1..8) {
            assertEquals(0L, timeline.frame(plan.end + (pass - 1) * timeline.period).source)
            assertEquals(plan.end - 1, timeline.frame(plan.end + pass * timeline.period - 1).source)
        }
    }

    @Test fun smallAuthoredPrerollLimitsCrossfadeInsteadOfShorteningPeriod() {
        val audio = source(start = 7, end = 38321)
        val plan = AudioExtender.plan(audio, region(audio), 500) { false }
        assertEquals(7, plan.overlap)
        val timeline = ExtensionTimeline(plan.start, plan.end, plan.overlap)
        assertEquals(38314L, timeline.period)
        assertEquals(0L, timeline.frame(plan.end - plan.overlap).incoming)
        assertEquals(7L, timeline.frame(plan.end).source)
        assertEquals(7L, timeline.frame(plan.end + timeline.period).source)
    }

    @Test fun manualCoordinatesOverrideEvenInvalidEmbeddedMetadata() {
        val audio = source(start = 999999, end = 1999999, metadataRate = 48000)
        val manual = LoopRegion(200, 1000, 1f, 1f, false, LoopKind.MANUAL)
        val plan = AudioExtender.plan(audio, manual, 20) { false }
        assertEquals(4410L, plan.start)
        assertEquals(22050L, plan.end)
        assertEquals(LoopKind.MANUAL, plan.kind)
    }

    @Test fun invalidEmbeddedCoordinatesOrSampleRateAreRejected() {
        for (audio in listOf(source(metadataRate = 48000), source(start = -1), source(end = 44101), source(start = 100, end = 100), source(start = 100, end = 99))) {
            try {
                AudioExtender.plan(audio, region(audio), 20) { false }
                fail("Invalid embedded coordinates must be rejected: ${audio.embeddedLoop}")
            } catch (_: IllegalArgumentException) { }
        }
        val missing = source().copy(embeddedLoop = null)
        try {
            AudioExtender.plan(missing, LoopRegion(0, 1000, 1f, 1f, false, LoopKind.EMBEDDED), 20) { false }
            fail("An embedded region requires source metadata")
        } catch (_: IllegalArgumentException) { }
    }

    @Test fun renderingAndStreamingMatchForExactEmbeddedPointsPrerollAndEndingFade() {
        for (channels in 1..2) for (start in listOf(0L, 7L, 1234L)) for (fade in listOf(0, 1)) {
            val audio = source(start = start, channels = channels)
            val region = region(audio)
            val output = temp.newFile().apply { delete() }
            AudioExtender.render(audio, region, 2500, fade, 100, output, { false }, {})
            val expected = output.readBytes()
            val plan = AudioExtender.plan(audio, region, 100) { false }
            ExtensionWaveReader(audio.file, plan, 2500, fade).use { reader ->
                assertEquals(expected.size.toLong(), reader.length)
                val actual = ByteArray(expected.size)
                var offset = 0
                while (offset < actual.size) {
                    val count = reader.read(offset.toLong(), actual, offset, minOf(997, actual.size - offset))
                    assertTrue(count > 0)
                    offset += count
                }
                assertArrayEquals("channels=$channels start=$start fade=$fade", expected, actual)
            }
        }
    }
}
