package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.random.Random

/** Musical form must survive candidate selection and the passages between alignment anchors. */
class DetectorStructureTest {
    private val rate=LoopAnalysis.RATE
    private val pitches=doubleArrayOf(93.0,123.0,164.0,219.0,292.0,389.0,519.0,692.0)
    private val notes=intArrayOf(1,5,2,7,0,4,6,3,5,0,7,2,4,1,6,0,3,7,5,2,6,4,0,1)

    private fun phrase(sequence:IntArray,phase:Double=0.0):FloatArray=FloatArray(sequence.size*rate/2) {frame->
        val within=(frame%(rate/2)).toDouble()/rate
        val time=frame.toDouble()/rate
        val frequency=pitches[sequence[frame/(rate/2)]]
        val envelope=(within/0.018).coerceAtMost(1.0)*((0.5-within)/0.035).coerceAtMost(1.0)
        ((0.24+0.055*sin(time*2.17)+0.025*sin(time*5.3))*envelope*
            (sin(2*PI*frequency*within+phase)+0.22*sin(2*PI*frequency*0.501*within-phase*0.7)+
                0.12*sin(2*PI*frequency*1.37*within+phase*0.4))).toFloat()
    }
    private fun intro():FloatArray {
        val random=Random(1297)
        return FloatArray(rate*3){(random.nextFloat()-0.5f)*0.13f}
    }
    private fun period(pcm:FloatArray,seconds:Double):LoopRegion {
        val messages=mutableListOf<String>()
        val loop=LoopAnalysis.detect(pcm,diagnostics=messages::add)
        assertNotNull("A verified complete form was rejected: ${messages.joinToString()}",loop)
        assertEquals("${messages.joinToString()}",LoopKind.REPEATED,loop!!.kind)
        assertEquals("${messages.joinToString()}",seconds,(loop.end-loop.start).toDouble()/rate,0.15)
        return loop
    }

    @Test fun contradictoryMiddleCannotBeHiddenByExactOpeningAndClosingRiffs() {
        val first=phrase(notes)
        val changed=notes.copyOf()
        for(i in 5 until 19)changed[i]=(changed[i]+3)%pitches.size
        val second=phrase(changed)
        // Keep exact PCM at both ends. The shared instrument, opening and closing
        // motifs do not establish that the intervening twelve-second form repeated.
        first.copyInto(second,0,0,rate*2)
        first.copyInto(second,first.size-rate*2,first.size-rate*2,first.size)
        val diagnostics=mutableListOf<String>()
        val result=LoopAnalysis.detect(intro()+first+second,diagnostics=diagnostics::add)
        assertNull("Contradictory bodies became an automatic loop: $result; ${diagnostics.joinToString()}",result)
    }

    @Test fun contradictoryRecurrenceCannotBecomeAGuessedWholeFileLoopViaItsClosingPhrase() {
        val first=phrase(notes)
        val changed=notes.map {(it+3)%pitches.size}.toIntArray()
        val second=phrase(changed)
        first.copyInto(second,0,0,rate*4)
        // This final phrase also matches the opening and would make a superficially
        // plausible near-whole-file INFERRED seam after the true recurrence is rejected.
        first.copyInto(second,second.size-rate*2,0,rate*2)
        val diagnostics=mutableListOf<String>()
        val result=LoopAnalysis.detect(intro()+first+second,diagnostics=diagnostics::add)
        assertNull("Single-pass fallback resurrected contradictory repeats: $result; ${diagnostics.joinToString()}",result)
        assertTrue("The public detector must retain the conflicting-form decision: ${diagnostics.joinToString()}",
            diagnostics.any {it=="Conflicting repeated forms rule out single-pass inference"})
    }

    @Test fun independentSectionsRetainTrueFormWithTransientVariation() {
        val first=phrase(notes)
        val second=first.copyOf()
        // A short percussion/transient difference is not a different musical body.
        val random=Random(744)
        for(i in rate*5 until rate*5+rate/12)second[i]+=(random.nextFloat()-0.5f)*0.035f
        period(intro()+first+second,12.0)
    }

    @Test fun threePhaseAlteredPassesAreNotForcedTowardATwoPassPeriod() {
        period(intro()+phrase(notes)+phrase(notes,1.2)+phrase(notes,0.7),12.0)
    }

    @Test fun twoPassesWithNoiseReductionAndALongFadeKeepTheirCompletePeriod() {
        val clean=phrase(notes)
        val random=Random(1324)
        val noisy=FloatArray(clean.size){clean[it]+(random.nextFloat()-0.5f)*0.025f}
        val reduced=FloatArray(clean.size){frame->
            clean[frame]*0.68f*((clean.size-frame).toFloat()/(rate*10)).coerceAtMost(1f)
        }
        val loop=period(intro()+noisy+reduced,12.0)
        assertTrue("The long final fade must remain recognized",loop.fading)
    }

    @Test fun threePassesWithPhaseGainAndFinalFadeKeepTheirFundamentalForm() {
        val first=phrase(notes)
        val second=phrase(notes,1.2).map {it*0.85f}.toFloatArray()
        val third=phrase(notes,0.7)
        for(frame in third.indices)third[frame]*=0.68f*((third.size-frame).toFloat()/(rate*8)).coerceAtMost(1f)
        period(intro()+first+second+third,12.0)
    }

    @Test fun completeObservedCycleCanHaveADistinctNonrepeatingOutro() {
        val first=phrase(notes)
        val ending=phrase(intArrayOf(7,7,4,2))
        period(intro()+first+first+ending,12.0)
    }

    @Test fun shortRestartStillKeepsItsLongCycleWithGainChange() {
        val first=phrase(notes)
        val restart=FloatArray(rate*3){first[it]*0.72f}
        val loop=period(intro()+first+restart,12.0)
        assertTrue(loop.repeats>1f)
        assertTrue(loop.start>=rate*2)
    }

    @Test fun shorterBackingMotifDoesNotReplaceALongerChangingMelody() {
        val melody=phrase(notes)
        val backing=phrase(intArrayOf(0,4,1,5,2,6))
        val body=FloatArray(melody.size){melody[it]*0.65f+backing[it%backing.size]*0.35f}
        period(intro()+body+body+body,12.0)
    }

    @Test fun lagCandidateBudgetStaysBoundedAcrossSeveralPeriodScales() {
        val body=phrase(notes+notes.reversedArray()+notes.map {(it+2)%8}.toIntArray())
        val diagnostics=mutableListOf<String>()
        val loop=LoopAnalysis.detect(intro()+body+body,diagnostics=diagnostics::add)
        assertNotNull(loop)
        assertTrue(abs(loop!!.end-loop.start-body.size)<rate/6)
        val candidateLine=diagnostics.first {it.startsWith("Candidate periods:")}
        val periods=candidateLine.substringAfter(":").split(',').filter {it.isNotBlank()}
            .map {it.trim().removeSuffix("s").toFloat()}
        assertTrue("Detailed analysis must have a hard cap: $candidateLine",periods.size<=32)
        assertTrue("Whole-form lag was crowded out: $candidateLine",periods.any {abs(it-body.size.toFloat()/rate)<0.15f})
    }

    @Test fun denoisedTwoPassTrackWithSharedRestsAndFadeRetainsFullLoop() {
        val base=phrase(notes)
        for(pause in listOf(rate*2 until rate*2+rate*2/5, rate*6 until rate*6+rate*2/5, rate*9 until rate*9+rate*2/5)) {
            for(i in pause)base[i]=0f
        }
        val second=FloatArray(base.size){frame->
            val fade=((base.size-frame).toFloat()/(rate*5)).coerceIn(0f,1f)
            val prev=base[(frame-1).coerceAtLeast(0)]
            (base[frame]*0.82f+prev*0.18f)*0.78f*fade
        }
        val loop=period(intro()+base+second,12.0)
        assertTrue("Final fade should be recognized on denoised track",loop.fading)
    }

    @Test fun briefMidPassFillBridgesToEarlyUnfadedSeamInsteadOfFadingSecondHalf() {
        val first=phrase(notes)
        val secondNotes=notes.copyOf()
        secondNotes[8]=(secondNotes[8]+3)%pitches.size
        secondNotes[9]=(secondNotes[9]+5)%pitches.size
        val second=phrase(secondNotes)
        val random=Random(912)
        for(i in 0 until rate*4)first[i]+=(random.nextFloat()-0.5f)*0.03f
        for(frame in second.indices) {
            val fade=((second.size-frame).toFloat()/(rate*4)).coerceIn(0f,1f)
            second[frame]*=fade
        }
        val pre=intro()
        val loop=period(pre+first+second,12.0)
        assertTrue("Seam must be placed near the start of Pass 1, not after the mid-pass fill: $loop",
            loop.start<=pre.size+rate*5/2)
        assertTrue("Loop end must remain in the unfaded body of Pass 2: $loop",
            loop.end<=pre.size+first.size+second.size-rate*3)
    }

    @Test fun threePassTrackWithDeepThirdPassFadeOutRetainsFundamentalPeriod() {
        val first=phrase(notes)
        val second=phrase(notes,0.4)
        val third=phrase(notes,0.9)
        for(frame in third.indices) {
            val fade=((third.size-frame).toFloat()/(rate*9)).coerceIn(0.03f,1f)
            third[frame]*=fade*fade
        }
        val loop=period(intro()+first+second+third,12.0)
        assertTrue("Three-pass track must report multiple repeats: $loop",loop.repeats>2f)
    }

    @Test fun halfSampleShiftedSecondPassWithFadeRetainsFullLoop() {
        val first=phrase(notes)
        val second=FloatArray(first.size){frame->
            val next=first[(frame+1).coerceAtMost(first.lastIndex)]
            val fade=((first.size-frame).toFloat()/(rate*4)).coerceIn(0f,1f)
            (first[frame]*0.5f+next*0.5f)*0.76f*fade
        }
        period(intro()+first+second,12.0)
    }

    @Test fun longSinglePassSongWithReturningBoundaryPhraseIsStillInferred() {
        val boundary=phrase(intArrayOf(7,1,4,0))
        val rng=Random(409)
        val seq=IntArray(88)
        for(idx in 1 until seq.size) {
            var next=rng.nextInt(pitches.size)
            if(next==seq[idx-1])next=(next+1)%pitches.size
            seq[idx]=next
        }
        val middle=phrase(seq)
        val pcm=boundary+middle+boundary
        val loop=LoopAnalysis.detect(pcm) ?: LoopAnalysis.detect(pcm,strict=false)
        assertNotNull("A long single-pass song with a returning boundary phrase must be inferred",loop)
        assertEquals(LoopKind.INFERRED,loop!!.kind)
        assertTrue("Inferred single-pass loop must preserve the full ~46s body: $loop",
            abs((loop.end-loop.start)-(boundary.size+middle.size))<=rate)
    }

    @Test fun longSinglePassSongWithShortRestartAtEndKeepsFullPeriod() {
        val longBody=phrase(notes+notes.reversedArray()+notes.map {(it+3)%pitches.size}.toIntArray())
        val restart=FloatArray(rate*3){longBody[it]*0.82f}
        val loop=period(intro()+longBody+restart,36.0)
        assertTrue(loop.repeats>1f)
    }
}
