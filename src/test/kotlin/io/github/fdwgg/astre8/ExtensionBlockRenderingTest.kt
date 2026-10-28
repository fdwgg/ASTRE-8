package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*
import kotlin.random.Random

class ExtensionBlockRenderingTest {
    @get:Rule val temp=TemporaryFolder()

    // Deliberately scalar, independent reference for the optimized block renderer.
    // Include extrema as well as noise so unity copies, rounding, and clipping must
    // agree exactly at crossfades, loop wraps, fades, and arbitrary byte boundaries.
    private fun expected(samples:ShortArray,plan:ExtensionPlan,targetMs:Long,fadeSeconds:Int,fadeOffsetFrames:Long=0):ByteArray {
        val frames=targetMs*plan.rate/1000
        val fade=min(fadeSeconds.coerceIn(0,30)*plan.rate.toLong(),frames/4)
        val timeline=ExtensionTimeline(plan.start,plan.end,plan.overlap)
        return ByteBuffer.allocate((frames*plan.channels*2).toInt()).order(ByteOrder.LITTLE_ENDIAN).apply {
            repeat(frames.toInt()) {frame->
                val mapping=timeline.frame(frame.toLong())
                val globalFrame=frame+fadeOffsetFrames
                val gain=if(fade>0 && globalFrame>=frames-fade)(frames-globalFrame-1).coerceAtLeast(0).toFloat()/fade else 1f
                repeat(plan.channels) {channel->
                    val a=samples[(mapping.source*plan.channels+channel).toInt()].toInt()
                    val mixed=if(mapping.incoming>=0) a*(1-mapping.mix)+samples[(mapping.incoming*plan.channels+channel).toInt()]*mapping.mix else a.toFloat()
                    putShort((mixed*gain).roundToInt().coerceIn(-32768,32767).toShort())
                }
            }
        }.array()
    }

    @Test fun blockCopiesAndMixedRunsPreserveEveryPcmByte() {
        val random=Random(412)
        for(channels in 1..2)for(start in listOf(0L,107L,1003L))for(overlap in listOf(0,80,211))for(fade in listOf(0,3)) {
            val samples=ShortArray(49_153*channels) {index->when(index%7){0->Short.MIN_VALUE;1->Short.MAX_VALUE;else->random.nextInt(-32768,32768).toShort()}}
            val file=temp.newFile()
            file.writeBytes(ByteBuffer.allocate(samples.size*2).order(ByteOrder.LITTLE_ENDIAN).apply {samples.forEach {putShort(it)}}.array())
            val plan=ExtensionPlan(8000,channels,start,49_153,overlap,1f,2f,LoopKind.MANUAL)
            val expected=expected(samples,plan,18_000,fade)
            ExtensionWaveReader(file,plan,18_000,fade).use {reader->
                val actual=ByteArray(expected.size);var position=0
                // Crosses both 128 KiB source block boundaries and loop boundaries.
                while(position<actual.size) {
                    val count=reader.read(44L+position,actual,position,min(actual.size-position,65_533))
                    assertTrue(count>0);position+=count
                }
                assertArrayEquals("channels=$channels start=$start overlap=$overlap fade=$fade",expected,actual)
                repeat(40) {
                    val at=random.nextInt(expected.size);val size=min(32_769,expected.size-at)
                    val slice=ByteArray(size+2){47}
                    assertEquals(size,reader.read(44L+at,slice,1,size))
                    assertArrayEquals(expected.copyOfRange(at,at+size),slice.copyOfRange(1,1+size))
                    assertEquals(47.toByte(),slice.first());assertEquals(47.toByte(),slice.last())
                }
            }
        }
    }

    @Test fun emptyAndPartialFramesRemainCorrectAroundEveryTransition() {
        val samples=ShortArray(1021*2) {if(it%2==0)Short.MIN_VALUE else Short.MAX_VALUE}
        val file=temp.newFile()
        file.writeBytes(ByteBuffer.allocate(samples.size*2).order(ByteOrder.LITTLE_ENDIAN).apply {samples.forEach {putShort(it)}}.array())
        val plan=ExtensionPlan(8000,2,151,1021,70,1f,2f,LoopKind.MANUAL)
        val expected=expected(samples,plan,1000,1)
        ExtensionWaveReader(file,plan,1000,1).use {reader->
            for(frame in listOf(0,950,951,1020,1021,1890,1891,5999,6000,7999))for(byte in 0..3) {
                val position=frame*4+byte;val count=min(9,expected.size-position)
                val actual=ByteArray(count)
                assertEquals(count,reader.read(44L+position,actual,0,count))
                assertArrayEquals(expected.copyOfRange(position,position+count),actual)
            }
            assertEquals(0,reader.read(0,ByteArray(0),0,0))
            assertEquals(-1,reader.read(reader.length,ByteArray(3),0,3))
        }
    }
    @Test fun shiftedFinalFadeMatchesScalarMixBeforeRoundingAndKeepsWaveHeader() {
        val random=Random(136)
        for(channels in 1..2)for(offset in listOf(0L,33333L,49153L)) {
            val samples=ShortArray(49_153*channels){random.nextInt(-32768,32768).toShort()}
            val file=temp.newFile()
            file.writeBytes(ByteBuffer.allocate(samples.size*2).order(ByteOrder.LITTLE_ENDIAN).apply {samples.forEach {putShort(it)}}.array())
            val plan=ExtensionPlan(8000,channels,107,49_153,211,1f,2f,LoopKind.MANUAL)
            val expected=expected(samples,plan,18_000,3,offset)
            val header=ByteArray(44)
            ExtensionWaveReader(file,plan,18_000,3).use {assertEquals(44,it.read(0,header,0,44))}
            ExtensionWaveReader(file,plan,18_000,3,fadeOffsetFrames=offset).use {reader->
                val actual=ByteArray(expected.size);var position=0
                while(position<actual.size) {
                    val count=reader.read(44L+position,actual,position,min(actual.size-position,8191))
                    assertTrue(count>0);position+=count
                }
                assertArrayEquals("channels=$channels offset=$offset",expected,actual)
                val actualHeader=ByteArray(44)
                assertEquals(44,reader.read(0,actualHeader,0,44));assertArrayEquals(header,actualHeader)
                assertEquals(expected.size+44L,reader.length)
            }
        }
    }
}
