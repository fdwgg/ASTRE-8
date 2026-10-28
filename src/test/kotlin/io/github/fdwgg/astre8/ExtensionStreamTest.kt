package io.github.fdwgg.astre8

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*
import kotlin.random.Random

class ExtensionStreamTest {
    @get:Rule val temp=TemporaryFolder()
    private fun source(channels:Int):DecodedAudio {
        val frames=8000;val data=ByteBuffer.allocate(frames*channels*2).order(ByteOrder.LITTLE_ENDIAN)
        repeat(frames) {frame->repeat(channels) {channel->data.putShort((sin(frame*2*PI*(110+channel*55)/8000)*18000).roundToInt().toShort())}}
        return DecodedAudio(temp.newFile(),8000,channels,frames.toLong(),FloatArray(0)).also {it.file.writeBytes(data.array())}
    }
    @Test fun streamingMatchesFullWavRenderingIncludingCrossfadesAndEndingFade() {
        for(channels in 1..2)for(start in listOf(0L,400L))for(fade in listOf(0,3)) {
            val audio=source(channels);val region=LoopRegion(start,1400,1f,2f,false,LoopKind.MANUAL)
            val out=File(temp.root,"reference-$channels-$start-$fade.wav")
            AudioExtender.render(audio,region,5000,fade,20,out,{false},{})
            val expected=out.readBytes();val plan=AudioExtender.plan(audio,region,20){false}
            ExtensionWaveReader(audio.file,plan,5000,fade).use {stream->
                assertEquals(expected.size.toLong(),stream.length)
                val actual=ByteArray(expected.size);var position=0
                while(position<actual.size) {
                    val count=stream.read(position.toLong(),actual,position,min(997,actual.size-position));assertTrue(count>0);position+=count
                }
                assertArrayEquals(expected,actual)
                assertEquals(-1,stream.read(stream.length,ByteArray(1),0,1))
            }
        }
    }
    @Test fun arbitraryByteSeeksAndPartialReadsMatchRenderedAudio() {
        val audio=source(2);val region=LoopRegion(400,1400,1f,2f,false,LoopKind.MANUAL);val out=File(temp.root,"seek-reference.wav")
        AudioExtender.render(audio,region,5000,2,20,out,{false},{})
        val expected=out.readBytes();val random=Random(21)
        ExtensionWaveReader(audio.file,AudioExtender.plan(audio,region,20){false},5000,2).use {stream->
            repeat(150) {
                val position=if(it<48)it else random.nextInt(expected.size)
                val size=random.nextInt(1,4001);val actual=ByteArray(size+6){91}
                val count=stream.read(position.toLong(),actual,3,size)
                assertEquals(min(size,expected.size-position),count)
                assertArrayEquals(expected.copyOfRange(position,position+count),actual.copyOfRange(3,3+count))
                assertEquals(91.toByte(),actual[2]);assertEquals(91.toByte(),actual[3+count])
            }
            assertEquals(0,stream.read(stream.length,ByteArray(0),0,0))
        }
    }
    @Test fun oneHourCanSeekDirectlyToEndingWithoutCreatingExtendedFile() {
        val audio=source(2);val plan=AudioExtender.plan(audio,LoopRegion(400,1400,1f,2f,false,LoopKind.MANUAL),20){false}
        val before=temp.root.listFiles()!!.map {it.name to it.length()}
        ExtensionWaveReader(audio.file,plan,3_600_000,5).use {stream->
            assertEquals(44L+3_600*8000*4,stream.length)
            val tail=ByteArray(12)
            assertEquals(12,stream.read(stream.length-12,tail,0,12))
            assertArrayEquals(ByteArray(4),tail.takeLast(4).toByteArray())
            assertTrue(stream.read(44+8000L*4*1800,ByteArray(4096),0,4096)>0)
        }
        assertEquals(before,temp.root.listFiles()!!.map {it.name to it.length()})
    }
    @Test fun incompleteSourceCannotProduceSilentlyCorruptPlayback() {
        val audio=source(1);val plan=ExtensionPlan(8000,1,1000,9000,100,1f,2f,LoopKind.REPEATED)
        assertThrows(IllegalArgumentException::class.java){ExtensionWaveReader(audio.file,plan,60_000,5)}
    }
    @Test fun invalidTargetIsRejectedBeforePlaybackCanReplaceTheOriginal() {
        val plan=ExtensionPlan(48000,2,1000,100000,100,1f,2f,LoopKind.REPEATED,48000L*60)
        assertThrows(IllegalArgumentException::class.java){plan.waveBytes(30_000)}
        val highRate=ExtensionPlan(384000,2,1000,100000,100,1f,2f,LoopKind.REPEATED,384000L*60)
        assertThrows(IllegalArgumentException::class.java){highRate.waveBytes(3_600_000)}
        assertEquals(44L+48000*2*2*60*60,plan.waveBytes(3_600_000))
    }
}
