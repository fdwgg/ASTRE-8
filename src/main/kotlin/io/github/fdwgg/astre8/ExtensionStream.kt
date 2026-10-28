package io.github.fdwgg.astre8

import java.io.Closeable
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/** Source-sized audio and a finite, seekable rendering recipe; independent of the chosen duration. */
data class ExtensionPlan(val rate:Int,val channels:Int,val start:Long,val end:Long,val overlap:Int,
    val confidence:Float,val repeats:Float,val kind:LoopKind,val sourceFrames:Long=end) {
    init { require(rate in 8000..384000 && channels in 1..2 && sourceFrames>=end);ExtensionTimeline(start,end,overlap) }
    val pcmBytes get()=end*channels*2
    fun sourcePositionMs(positionMs:Long)=ExtensionTimeline(start,end,overlap).frame(positionMs.coerceAtLeast(0)*rate/1000).source*1000/rate
    fun waveBytes(targetMs:Long):Long {
        val frames=targetMs*rate/1000
        require(frames>sourceFrames) {"Choose a duration longer than the original track."}
        val bytes=frames*channels*2
        require(bytes<Int.MAX_VALUE-44) {"This source format exceeds the WAV size limit at this duration. Choose a shorter extension."}
        return 44+bytes
    }
}

/** Renders only bytes requested by the player's loader. No growing files, polling, or full-song allocation. */
class ExtensionWaveReader(file:File,val plan:ExtensionPlan,targetMs:Long,fadeSeconds:Int,
    private val fadeOffsetFrames:Long=0):Closeable {
    private val totalFrames=targetMs*plan.rate/1000
    private val frameBytes=plan.channels*2
    val length=plan.waveBytes(targetMs)
    private val dataBytes=length-44
    private val timeline=ExtensionTimeline(plan.start,plan.end,plan.overlap)
    private val fadeFrames=min(fadeSeconds.coerceIn(0,30)*plan.rate.toLong(),totalFrames/4)
    // An adaptive handoff can retain already-heard original repeats before this
    // renderer's loop phase. Fade against that global output timeline, while
    // keeping source mapping and the finite WAV header unchanged.
    private val fadeStart=totalFrames-fadeFrames-fadeOffsetFrames
    private val primary:Samples
    private val incoming:Samples
    private val frameBuffer=ByteArray(frameBytes)
    private var cachedFrame=-1L
    private val header:ByteArray
    init {
        require(fadeOffsetFrames>=0 && fadeOffsetFrames<totalFrames) {"Fade offset is outside the extension timeline."}
        require(file.length()>=plan.pcmBytes) {"Cached source audio is incomplete."}
        primary=Samples(RandomAccessFile(file,"r"),plan.channels)
        try {incoming=Samples(RandomAccessFile(file,"r"),plan.channels)} catch(error:Exception) {primary.close();throw error}
        header=ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray());putInt(dataBytes.toInt()+36);put("WAVEfmt ".toByteArray());putInt(16)
            putShort(1);putShort(plan.channels.toShort());putInt(plan.rate);putInt(plan.rate*frameBytes)
            putShort(frameBytes.toShort());putShort(16);put("data".toByteArray());putInt(dataBytes.toInt())
        }.array()
    }
    fun read(position:Long,buffer:ByteArray,offset:Int,length:Int):Int {
        require(position>=0 && offset>=0 && length>=0 && offset<=buffer.size-length)
        if(length==0)return 0
        if(position>=this.length)return -1
        var cursor=position;var dest=offset
        val stop=offset+min(length.toLong(),this.length-position).toInt()
        if(cursor<44) {
            val count=min(44-cursor, (stop-dest).toLong()).toInt()
            header.copyInto(buffer,dest,cursor.toInt(),cursor.toInt()+count);cursor+=count;dest+=count
        }
        while(dest<stop) {
            val frame=(cursor-44)/frameBytes
            val within=((cursor-44)%frameBytes).toInt()
            if(within!=0 || stop-dest<frameBytes) {
                // Byte-range requests can split a PCM frame. Keep only that tiny edge
                // frame; the aligned body below bypasses per-sample rendering entirely.
                if(frame!=cachedFrame) {renderFrames(frame,timeline.sourceFrame(frame),1,frameBuffer,0);cachedFrame=frame}
                val count=min(frameBytes-within,stop-dest)
                frameBuffer.copyInto(buffer,dest,within,within+count);dest+=count;cursor+=count
                continue
            }
            val source=timeline.sourceFrame(frame)
            val incomingFrame=timeline.incomingFrame(source)
            var frames=min((stop-dest)/frameBytes,(plan.end-source).coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            // A run ends at a loop, crossfade, or final-fade boundary. Mapping the
            // timeline once per run avoids long division/modulo on every sample.
            if(incomingFrame<0 && plan.overlap>0)frames=min(frames,(plan.end-plan.overlap-source).coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            if(fadeFrames>0 && frame<fadeStart)frames=min(frames,(fadeStart-frame).coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
            if(incomingFrame<0 && (fadeFrames==0L || frame<fadeStart)) {
                // Unity gain PCM is already exactly the required output, including
                // negative extrema. Copy from the existing bounded source block cache.
                primary.copyFrames(source,frames,buffer,dest)
            } else renderFrames(frame,source,frames,buffer,dest)
            val count=frames*frameBytes;dest+=count;cursor+=count
        }
        return dest-offset
    }
    private fun renderFrames(outputFrame:Long,sourceFrame:Long,frames:Int,buffer:ByteArray,offset:Int) {
        val firstIncoming=timeline.incomingFrame(sourceFrame)
        var dest=offset
        repeat(frames) {index->
            val source=sourceFrame+index
            val incomingFrame=if(firstIncoming>=0)firstIncoming+index else -1L
            val mix=timeline.mix(source)
            val frame=outputFrame+index
            // Mix and apply gain in float before the one and only PCM rounding.
            // Bytes after the shifted virtual ending remain silent, never inverted.
            val fade=if(fadeFrames>0 && frame>=fadeStart)(totalFrames-frame-fadeOffsetFrames-1).coerceAtLeast(0).toFloat()/fadeFrames else 1f
            for(channel in 0 until plan.channels) {
                val a=primary.sample(source,channel)
                val value=((if(incomingFrame>=0)a*(1-mix)+incoming.sample(incomingFrame,channel)*mix else a.toFloat())*fade)
                    .roundToInt().coerceIn(-32768,32767)
                buffer[dest++]=value.toByte();buffer[dest++]=(value shr 8).toByte()
            }
        }
    }
    override fun close() {try {primary.close()} finally {incoming.close()}}
    private class Samples(private val file:RandomAccessFile,private val channels:Int):Closeable {
        private val bytes=ByteArray(128*1024);private var base=-1L;private var count=0
        private fun fill(position:Long) {
            base=position;file.seek(base);count=file.read(bytes)
            check(count>=2){"Unexpected end of cached source audio."}
        }
        fun copyFrames(frame:Long,frames:Int,target:ByteArray,targetOffset:Int) {
            var position=frame*channels*2;var dest=targetOffset;val stop=targetOffset+frames*channels*2
            while(dest<stop) {
                if(position<base || position>=base+count)fill(position)
                val offset=(position-base).toInt();val size=min(stop-dest,count-offset)
                bytes.copyInto(target,dest,offset,offset+size);position+=size;dest+=size
            }
        }
        fun sample(frame:Long,channel:Int):Int {
            val position=(frame*channels+channel)*2
            if(position<base || position+1>=base+count)fill(position)
            val offset=(position-base).toInt()
            return ((bytes[offset].toInt() and 255) or (bytes[offset+1].toInt() shl 8)).toShort().toInt()
        }
        override fun close()=file.close()
    }
}
