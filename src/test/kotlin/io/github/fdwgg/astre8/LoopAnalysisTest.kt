package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CancellationException
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/** Phrase-level fixtures: matching one riff or one terminal note is not a whole song loop. */
class LoopAnalysisTest {
    private val rate = LoopAnalysis.RATE
    private val pitches = doubleArrayOf(93.0, 123.0, 164.0, 219.0, 292.0, 389.0, 519.0, 692.0)
    // No repeated adjacent pair, despite using the same instrument and pitch vocabulary throughout.
    private val throughNotes = intArrayOf(
        0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7,
        1, 4, 7, 2, 5, 0, 3, 6, 2, 6, 3, 7, 4, 0, 5, 1
    )
    private val bodyNotes = intArrayOf(
        1, 5, 2, 7, 0, 4, 6, 3, 5, 0, 7, 2,
        4, 1, 6, 0, 3, 7, 5, 2, 6, 4, 0, 1
    )

    private fun phrase(notes: IntArray, phase: Double = 0.0, noteSeconds: Double = 0.5): FloatArray {
        val noteFrames = (rate * noteSeconds).toInt()
        return FloatArray(notes.size * noteFrames) { frame ->
            val note = frame / noteFrames
            val within = (frame % noteFrames).toDouble() / rate
            val time = frame.toDouble() / rate
            val frequency = pitches[notes[note]]
            val attack = (within / 0.018).coerceAtMost(1.0)
            val release = ((noteSeconds - within) / 0.035).coerceAtMost(1.0)
            val movement = 0.24 + 0.055 * sin(time * 2.17) + 0.025 * sin(time * 5.3)
            // Identical spectral instrument, clear pitch changes, and a varied rhythmic envelope.
            val carrier = sin(2 * PI * frequency * within + phase) +
                0.22 * sin(2 * PI * frequency * 0.501 * within - phase * 0.7) +
                0.12 * sin(2 * PI * frequency * 1.37 * within + phase * 0.4)
            (movement * attack * release * carrier).toFloat()
        }
    }

    private fun intro(seconds: Int = 3): FloatArray {
        val random = Random(1297)
        return FloatArray(rate * seconds) { (random.nextFloat() - 0.5f) * 0.13f }
    }

    private fun uniqueNotePairs(count: Int): IntArray {
        // An Euler path through the directed pitch pairs avoids accidentally making a second
        // copy of the whole background while constructing the isolated-riff regression.
        val random = Random(751)
        val available = Array(pitches.size) { from -> pitches.indices.filter { it != from }.shuffled(random).toMutableList() }
        val stack = mutableListOf(0)
        val path = mutableListOf<Int>()
        while (stack.isNotEmpty()) {
            val current = stack.last()
            if (available[current].isNotEmpty()) stack += available[current].removeAt(available[current].lastIndex)
            else path += stack.removeAt(stack.lastIndex)
        }
        return path.asReversed().take(count).toIntArray()
    }

    private fun assertPeriod(pcm: FloatArray, expectedSeconds: Double, toleranceSeconds: Double = 0.10): LoopRegion {
        val diagnostics = mutableListOf<String>()
        val loop = requireNotNull(LoopAnalysis.detect(pcm, diagnostics = diagnostics::add)) { "A full repeated musical body was rejected: ${diagnostics.joinToString()}" }
        assertEquals("Wrong evidence type: $loop. ${diagnostics.joinToString()}", LoopKind.REPEATED, loop.kind)
        val actual = (loop.end - loop.start).toDouble() / rate
        assertTrue("Expected ${expectedSeconds}s complete body, got ${actual}s: $loop", abs(actual - expectedSeconds) <= toleranceSeconds)
        assertTrue("The detected boundary must stay inside the source", loop.start >= 0 && loop.end < pcm.size)
        return loop
    }

    @Test fun isolatedExactRiffDoesNotTurnAnOtherwiseThroughComposedSongIntoALoop() {
        val notes = uniqueNotePairs(56)
        // Two exact copies of a four-second riff amid unrelated surrounding musical phrases.
        val motif = intArrayOf(6, 0, 5, 2, 7, 3, 1, 4)
        motif.copyInto(notes, 8)
        motif.copyInto(notes, 40)
        val pcm = phrase(notes)
        // Copy the rendered samples too: no accidental phase/envelope mismatch can hide this trap.
        pcm.copyInto(pcm, 20 * rate, 4 * rate, 8 * rate)
        assertNull("A locally repeated riff is not evidence that the intervening 16-second body repeats", LoopAnalysis.detect(pcm))
        assertNull("Relaxed mode must still reject a local motif as a full-song loop", LoopAnalysis.detect(pcm, strict = false))
    }

    @Test fun sameTimbreAndTerminalPitchWithDifferentNoteOrderDoesNotInventSinglePassBoundary() {
        val pcm = phrase(throughNotes)
        assertNull("Unrelated note order cannot be bridged merely because the instrument and terminal pitch recur", LoopAnalysis.detect(pcm))
        assertNull("Relaxing seam precision must not discard musical phrase validation", LoopAnalysis.detect(pcm, strict = false))
    }

    @Test fun sameClosingPhraseWithALargeLevelJumpDoesNotBecomeAnAutomaticSeam() {
        val first = phrase(intArrayOf(0, 3, 6, 1))
        val middle = phrase(intArrayOf(2, 7, 4, 0, 5, 2, 6, 3, 7, 1, 4, 2, 5, 0, 6, 4, 7, 3, 1, 5, 6, 2, 0, 7))
        val quietEnding = FloatArray(first.size) { first[it] * 0.055f }
        val pcm = first + middle + quietEnding
        assertNull("A nearly inaudible ending must not jump abruptly to the loud opening phrase", LoopAnalysis.detect(pcm))
    }

    @Test fun twoCompletePassesRetainTheFullPeriodDespiteIntroGainChangeAndFade() {
        val body = phrase(bodyNotes)
        val second = FloatArray(body.size) { frame ->
            val fade = ((body.size - frame).toFloat() / (rate * 2)).coerceAtMost(1f)
            body[frame] * 0.62f * fade
        }
        val loop = assertPeriod(intro() + body + second, 12.0)
        assertTrue("The final decay should be recognized", loop.fading)
    }

    @Test fun phaseShiftedSecondPassRetainsMusicalPeriodInsteadOfPickingATinyRiff() {
        val body = phrase(bodyNotes)
        val second = phrase(bodyNotes, phase = 1.2)
        assertPeriod(intro() + body + second, 12.0, toleranceSeconds = 0.15)
    }

    @Test fun oneCompletePassWithOnlyAThreeSecondRestartStillHasObservedLoopEvidence() {
        val body = phrase(bodyNotes)
        val loop = assertPeriod(intro() + body + body.copyOf(rate * 3), 12.0)
        assertTrue("Short restart must remain an observed repetition", loop.repeats > 1f)
        assertTrue("The intro must stay outside the repeated body", loop.start >= rate * 2)
    }

    @Test fun matchingTwoSecondBoundaryPhraseSupportsAnExplicitlyInferredSinglePass() {
        val boundary = phrase(intArrayOf(7, 1, 4, 0))
        val middle = phrase(intArrayOf(2, 6, 3, 5, 1, 7, 2, 4, 6, 0, 5, 3, 7, 6, 1, 2, 0, 4, 3, 6))
        val pcm = boundary + middle + boundary
        val loop = LoopAnalysis.detect(pcm)
        assertNotNull("A whole matching boundary phrase should still allow cautious single-pass inference", loop)
        assertEquals("A short closing phrase does not prove the entire body repeated", LoopKind.INFERRED, loop!!.kind)
        assertTrue("Inferred loop should preserve most of the musical body", loop.end - loop.start >= rate * 10)
    }

    @Test fun inferredBoundaryPhraseDoesNotRequirePeriodToLandOnTheFiftyMillisecondFeatureGrid() {
        val boundary = phrase(intArrayOf(7, 1, 4, 0))
        val middle = phrase(intArrayOf(2, 6, 3, 5, 1, 7, 2, 4, 6, 0, 5, 3, 7, 6, 1, 2, 0, 4, 3, 6))
        val insertion = middle.size / 2
        val extraFrames = rate * 13 / 1000
        // Duplicate 13 ms inside the unrelated middle. The closing phrase is still an exact
        // waveform repeat, but its true recurrence now falls between the 50 ms FFT hops.
        val irregularMiddle = middle.copyOfRange(0, insertion) +
            middle.copyOfRange(insertion, insertion + extraFrames) + middle.copyOfRange(insertion, middle.size)
        val expectedPeriod = boundary.size + irregularMiddle.size
        val loop = LoopAnalysis.detect(boundary + irregularMiddle + boundary)
        assertNotNull("A real matching phrase must not disappear because its period is off the feature grid", loop)
        assertEquals(LoopKind.INFERRED, loop!!.kind)
        assertTrue("The inferred period should preserve the complete musical body: $loop", abs(loop.end - loop.start - expectedPeriod) <= rate / 40)
    }

    @Test fun cancellationStopsAnalysisBothBeforeAndDuringFeatureExtraction() {
        val body = phrase(bodyNotes)
        val pcm = intro() + body + body
        assertThrows(CancellationException::class.java) { LoopAnalysis.detect(pcm, cancelled = { true }) }
        var checks = 0
        assertThrows(CancellationException::class.java) { LoopAnalysis.detect(pcm, cancelled = { ++checks >= 5 }) }
        assertEquals("Cancellation should stop as soon as the predicate changes", 5, checks)
    }
}
