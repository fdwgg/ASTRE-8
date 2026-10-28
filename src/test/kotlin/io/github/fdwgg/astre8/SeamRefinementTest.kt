package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

class SeamRefinementTest {
    private fun withSource(shift:Int,check:(DecodedAudio,Long)->Unit) {
        val rate=8000;val start=rate;val end=rate*5;val trueEnd=end+shift
        val random=Random(174)
        val samples=ShortArray(rate*7*2){random.nextInt(-18000,18000).toShort()}
        // Only the pre-boundary stereo context repeats. The audio immediately after the
        // boundary is unrelated, so optimizing post-boundary audio would miss the seam.
        for(frame in 1..rate/4)for(channel in 0..1)
            samples[(trueEnd-frame)*2+channel]=samples[(start-frame)*2+channel]
        val bytes=ByteBuffer.allocate(samples.size*2).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach(bytes::putShort)
        val file=File.createTempFile("muzbox-seam-",".pcm")
        try {
            file.writeBytes(bytes.array())
            check(DecodedAudio(file,rate,2,(samples.size/2).toLong(),FloatArray(0)),trueEnd.toLong())
        } finally {file.delete()}
    }

    @Test fun refinementMatchesStereoPrerollRatherThanUnrelatedAudioAfterTheJoin()=withSource(4) {audio,trueEnd->
        val region=LoopRegion(2000,10000,0.98f,2f,false)
        val plan=AudioExtender.plan(audio,region,80){false}
        assertEquals(trueEnd,plan.end)
        assertEquals(8000L,plan.start)
    }

    @Test fun inferredQuantizedBoundaryCanAlignAcrossTwentyMilliseconds()=withSource(80) {audio,trueEnd->
        val region=LoopRegion(2000,10000,0.85f,1f,false,LoopKind.INFERRED,120)
        assertEquals(trueEnd,AudioExtender.plan(audio,region,80){false}.end)
    }

    @Test fun explicitManualPointsAreNeverShiftedByAutomaticRefinement()=withSource(4) {audio,_->
        val region=LoopRegion(2000,10000,0.9f,1f,false,LoopKind.MANUAL)
        assertEquals(40000L,AudioExtender.plan(audio,region,80){false}.end)
    }
}
