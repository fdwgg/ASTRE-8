package io.github.fdwgg.astre8

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.*
import java.nio.ByteOrder
import java.util.concurrent.CancellationException
import kotlin.math.*

data class DecodedAudio(val file: File, val rate: Int, val channels: Int, val frames: Long, val analysis: FloatArray,
    val embeddedLoop: EmbeddedLoop? = null)
data class ExtendedAudio(val file: File, val startMs: Long, val endMs: Long, val confidence: Float, val repeats: Float, val kind: LoopKind = LoopKind.REPEATED)

object AudioExtender {
    const val VERSION = 8
    fun decode(context: Context, uri: Uri, out: File, cancelled: () -> Boolean, progress: (String) -> Unit, reserve:(Long)->Unit={}, wave:Boolean=false): DecodedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val track = (0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true }
                ?: error("This file contains no supported audio stream.")
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val lengthUs = if (format.containsKey(MediaFormat.KEY_DURATION)) format.getLong(MediaFormat.KEY_DURATION) else 0L
            require(lengthUs <= 20L * 60 * 1_000_000) { "Analysis supports source tracks up to 20 minutes." }
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var sourceChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var channels = min(2, sourceChannels)
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            if(wave && format.containsKey(MediaFormat.KEY_PCM_ENCODING)) require(format.getInteger(MediaFormat.KEY_PCM_ENCODING)==encoding) {"High-resolution audio uses its original decoder."}
            format.setInteger(MediaFormat.KEY_PCM_ENCODING, encoding)
            // This decoder prepares source PCM while another track can be playing.
            // Let the platform schedule it below the realtime playback decoder.
            format.setInteger(MediaFormat.KEY_PRIORITY,1)
            if(wave) {
                require(sourceChannels in 1..2) {"Multichannel audio uses its original decoder."}
            }
            val decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec = decoder; decoder.configure(format, null, null, 0); decoder.start()
            val info = MediaCodec.BufferInfo()
            var inputDone = false; var outputDone = false; var frames = 0L
            val maxAnalysisSamples=LoopAnalysis.RATE*20*60+1
            // Allocate for the source duration rather than reserving twenty minutes
            // for every song. Unknown/misreported durations grow only as needed,
            // under the same hard analysis limit.
            val expectedSamples=if(lengthUs>0)lengthUs/1_000_000*LoopAnalysis.RATE+(lengthUs%1_000_000)*LoopAnalysis.RATE/1_000_000+LoopAnalysis.RATE+1 else LoopAnalysis.RATE*30L
            var analysis=FloatArray(if(wave)0 else expectedSamples.coerceIn(LoopAnalysis.RATE.toLong(),maxAnalysisSamples.toLong()).toInt())
            var analysisCount = 0
            var sum = 0.0; var count = 0; var nextAnalysisFrame = rate.toDouble() / LoopAnalysis.RATE
            var lastOutput = System.nanoTime()
            // Reuse one bounded PCM block instead of allocating an array for every
            // codec output buffer (or expanding to the codec's largest output).
            val outputBlock=ByteArray(64*1024)
            BufferedOutputStream(FileOutputStream(out), 128*1024).use { output ->
                if(wave) {reserve(44);output.write(ByteArray(44))}
                while (!outputDone) {
                    if (cancelled()) throw CancellationException()
                    if (System.nanoTime() - lastOutput > 30_000_000_000L) error("Audio decoder stalled. Try another source format.")
                    if (!inputDone) {
                        val index = decoder.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buffer = decoder.getInputBuffer(index)!!
                            val size = extractor.readSampleData(buffer, 0)
                            if (size < 0) { decoder.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone=true }
                            else { decoder.queueInputBuffer(index, 0, size, extractor.sampleTime, 0); extractor.advance() }
                        }
                    }
                    when (val index = decoder.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            val actual = decoder.outputFormat
                            if(wave && frames>0)require(rate==actual.getInteger(MediaFormat.KEY_SAMPLE_RATE) && sourceChannels==actual.getInteger(MediaFormat.KEY_CHANNEL_COUNT)) {"Changing audio formats use their original decoder."}
                            rate = actual.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                            sourceChannels = actual.getInteger(MediaFormat.KEY_CHANNEL_COUNT); channels = min(2, sourceChannels)
                            encoding = if (actual.containsKey(MediaFormat.KEY_PCM_ENCODING)) actual.getInteger(MediaFormat.KEY_PCM_ENCODING) else AudioFormat.ENCODING_PCM_16BIT
                            if(wave) require(sourceChannels in 1..2 && encoding==AudioFormat.ENCODING_PCM_16BIT) {"This audio format uses its original decoder to preserve quality."}
                            nextAnalysisFrame = rate.toDouble() / LoopAnalysis.RATE
                        }
                        else -> if (index >= 0) {
                            lastOutput = System.nanoTime()
                            val buffer = decoder.getOutputBuffer(index)!!.order(ByteOrder.LITTLE_ENDIAN)
                            buffer.position(info.offset); buffer.limit(info.offset + info.size)
                            if(wave) {
                                // Preserve the decoder's PCM bytes, including its gapless trimming.
                                val size=buffer.remaining()
                                require(size%(channels*2)==0)
                                frames+=size/(channels*2)
                                reserve(44+frames*channels*2)
                                while(buffer.hasRemaining()) {
                                    val sizeToWrite=min(buffer.remaining(),outputBlock.size)
                                    buffer.get(outputBlock,0,sizeToWrite);output.write(outputBlock,0,sizeToWrite)
                                }
                                outputDone=info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM!=0
                                decoder.releaseOutputBuffer(index,false)
                                require(frames<=rate*20L*60 && frames*channels*2<=512L*1024*1024)
                                continue
                            }
                            val bytes = when (encoding) { AudioFormat.ENCODING_PCM_FLOAT, AudioFormat.ENCODING_PCM_32BIT -> 4; AudioFormat.ENCODING_PCM_24BIT_PACKED -> 3; AudioFormat.ENCODING_PCM_8BIT -> 1; else -> 2 }
                            var p = 0
                            while (buffer.remaining() >= bytes*sourceChannels) {
                                var left = 0f; var right = 0f; var lc = 0; var rc = 0
                                if(encoding==AudioFormat.ENCODING_PCM_16BIT && sourceChannels<=2) {
                                    // The usual mono/stereo path has no channel mixing.
                                    // Keep exactly the same normalization/rounding as
                                    // existing cached sources and analysis samples.
                                    left=buffer.short/32768f
                                    right=if(sourceChannels==2)buffer.short/32768f else left
                                } else {
                                    for (c in 0 until sourceChannels) {
                                        val sample = when (encoding) {
                                            AudioFormat.ENCODING_PCM_FLOAT -> buffer.float
                                            AudioFormat.ENCODING_PCM_32BIT -> buffer.int / 2147483648f
                                            AudioFormat.ENCODING_PCM_24BIT_PACKED -> { val v=(buffer.get().toInt() and 255) or ((buffer.get().toInt() and 255) shl 8) or (buffer.get().toInt() shl 16); v/8388608f }
                                            AudioFormat.ENCODING_PCM_8BIT -> ((buffer.get().toInt() and 255)-128)/128f
                                            else -> buffer.short / 32768f
                                        }
                                        if (c%2==0) { left+=sample; lc++ } else { right+=sample; rc++ }
                                    }
                                    left/=lc.coerceAtLeast(1); right=if(rc>0) right/rc else left
                                }
                                for (c in 0 until channels) {
                                    val value = ((if(c==0) left else right).coerceIn(-1f,1f)*32767).roundToInt()
                                    outputBlock[p++]=value.toByte(); outputBlock[p++]=(value shr 8).toByte()
                                }
                                sum += left*0.6f+right*0.4f; count++; frames++
                                if (frames >= nextAnalysisFrame) {
                                    if(analysisCount>=analysis.size) {
                                        if(analysisCount>=maxAnalysisSamples)error("Source exceeds the 20 minute analysis limit.")
                                        analysis=analysis.copyOf(min(maxAnalysisSamples,analysis.size*2))
                                    }
                                    analysis[analysisCount++]=(sum/count).toFloat(); sum=0.0; count=0
                                    nextAnalysisFrame += rate.toDouble()/LoopAnalysis.RATE
                                }
                                if(p==outputBlock.size) {
                                    reserve(frames*channels*2);output.write(outputBlock,0,p);p=0
                                }
                            }
                            reserve(frames*channels*2)
                            output.write(outputBlock, 0, p)
                            if (frames > rate * 20L * 60 || frames*channels*2 > 512L*1024*1024) error("Source exceeds the 20 minute / 512 MB analysis limit.")
                            if (frames % rate < 4096) progress("Decoding · ${frames / rate}s")
                            outputDone = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            decoder.releaseOutputBuffer(index, false)
                        }
                    }
                }
            }
            if(wave) {
                require(frames>0) {"The decoder produced no audio."}
                val header=ByteArrayOutputStream(44).also {writeWavHeader(it,(frames*channels*2).toInt(),rate,channels)}.toByteArray()
                RandomAccessFile(out,"rw").use {it.write(header)}
            }
            return DecodedAudio(out, rate, channels, frames, analysis.copyOf(analysisCount))
        } finally {
            runCatching {codec?.stop()};runCatching {codec?.release()};runCatching {extractor.release()}
        }
    }

    private data class LoopPoints(val start:Long,val end:Long,val overlap:Int)

    private fun points(audio:DecodedAudio,region:LoopRegion,crossfadeMs:Int,cancelled:()->Boolean):LoopPoints {
        val authored=if(region.kind==LoopKind.EMBEDDED)requireNotNull(audio.embeddedLoop) {"Source loop points are unavailable."} else null
        require(authored==null || authored.rate==audio.rate) {"Source loop rate differs from decoded audio."}
        val start=authored?.startFrame ?: (region.start*audio.rate/LoopAnalysis.RATE)
        var end=authored?.endFrame ?: (region.end*audio.rate/LoopAnalysis.RATE)
        require(start>=0 && end<=audio.frames && end>start) {"Loop points are outside the source."}
        if(authored==null && region.confidence<1f && region.kind!=LoopKind.MANUAL) end=refineSeam(audio,start,end,region,cancelled)
        var overlap=min(max(crossfadeMs,region.suggestedCrossfadeMs)*audio.rate/1000,((end-start)/8).toInt())
        // Authored sample coordinates keep their exact period, including a zero-start loop.
        if(authored!=null)overlap=min(overlap,start.coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
        return LoopPoints(start,end,overlap)
    }

    fun plan(audio:DecodedAudio,region:LoopRegion,crossfadeMs:Int,cancelled:()->Boolean):ExtensionPlan {
        val points=points(audio,region,crossfadeMs,cancelled)
        return ExtensionPlan(audio.rate,audio.channels,points.start,points.end,points.overlap,region.confidence,region.repeats,region.kind,audio.frames)
    }

    fun render(audio: DecodedAudio, region: LoopRegion, targetMs: Long, fadeSeconds: Int, crossfadeMs: Int,
               out: File, cancelled: () -> Boolean, progress: (String) -> Unit): ExtendedAudio {
        val prepared=points(audio,region,crossfadeMs,cancelled)
        val start=prepared.start
        val end=prepared.end
        val totalFrames = targetMs * audio.rate / 1000
        require(totalFrames > audio.frames) { "Choose a duration longer than the original track." }
        val byteCount = totalFrames * audio.channels * 2
        require(byteCount < Int.MAX_VALUE - 44) { "The selected duration would create a file larger than 2 GB." }
        require(out.parentFile!!.usableSpace > byteCount + 16*1024*1024) { "Not enough free space to render this duration." }
        val overlap=prepared.overlap
        val timeline = ExtensionTimeline(start, end, overlap)
        val part = File(out.path + ".part")
        try {
            RandomAccessFile(audio.file, "r").use { source ->
                // Two independent block caches avoid disk seeks for each crossfade sample.
                val primary = PcmReader(source, audio.channels)
                RandomAccessFile(audio.file, "r").use { second ->
                    val incoming = PcmReader(second, audio.channels)
                    BufferedOutputStream(FileOutputStream(part), 128*1024).use { stream ->
                        writeWavHeader(stream, byteCount.toInt(), audio.rate, audio.channels)
                        val buffer = ByteArray(4096 * audio.channels * 2)
                        var frame = 0L
                        val fadeFrames = min(fadeSeconds * audio.rate.toLong(), totalFrames / 4)
                        while (frame < totalFrames) {
                            if (cancelled()) throw CancellationException()
                            val count = min(4096L, totalFrames-frame).toInt(); var p=0
                            repeat(count) {
                                val mapping = timeline.frame(frame)
                                val fade = if (fadeFrames>0 && frame >= totalFrames-fadeFrames) (totalFrames-frame-1).toFloat()/fadeFrames else 1f
                                for (channel in 0 until audio.channels) {
                                    val a = primary.sample(mapping.source, channel)
                                    // Linear constant-gain crossfade avoids gain bumps for correlated repeats.
                                    val mixed = if(mapping.incoming>=0) a*(1-mapping.mix)+incoming.sample(mapping.incoming, channel)*mapping.mix else a.toFloat()
                                    val value = (mixed*fade).roundToInt().coerceIn(-32768,32767)
                                    buffer[p++]=value.toByte(); buffer[p++]=(value shr 8).toByte()
                                }
                                frame++
                            }
                            stream.write(buffer,0,p)
                            if(frame % (audio.rate*2) < 4096) progress("Rendering · ${(frame*100/totalFrames)}%")
                        }
                    }
                }
            }
            check(part.renameTo(out)) { "Could not finalize the extended audio file." }
            return ExtendedAudio(out, start*1000/audio.rate, end*1000/audio.rate, region.confidence, region.repeats, region.kind)
        } finally { part.delete() }
    }

    private class PcmReader(val file: RandomAccessFile, val channels: Int,bufferBytes:Int=128*1024) {
        val bytes = ByteArray(bufferBytes); var base = -1L; var count = 0
        fun sample(frame: Long, channel: Int): Int {
            val position = (frame*channels+channel)*2
            if (position < base || position+1 >= base+count) {
                base=position; file.seek(base); count=file.read(bytes)
                check(count>=2) { "Unexpected end of decoded audio." }
            }
            val offset=(position-base).toInt()
            return ((bytes[offset].toInt() and 255) or (bytes[offset+1].toInt() shl 8)).toShort().toInt()
        }
    }
    private fun refineSeam(audio: DecodedAudio, start:Long, end:Long, region:LoopRegion, cancelled:()->Boolean):Long {
        // The stream crossfades the history BEFORE its boundaries, then resumes at start.
        // Matching a short transient after end can improve correlation while worsening
        // the audible join. Check the same pre-context in both original stereo channels.
        val width=min(audio.rate*12/100,start.toInt())
        if(width<audio.rate/50)return end
        RandomAccessFile(audio.file,"r").use {file ->
            RandomAccessFile(audio.file,"r").use {other ->
                val radius=(audio.rate/(if(region.kind==LoopKind.INFERRED || region.suggestedCrossfadeMs>0)50 else 500)).coerceAtLeast(1)
                // Cover the entire search window even for high sample rates, so each
                // candidate reuses the same blocks instead of rereading PCM from storage.
                val cacheBytes=max(128*1024,(width+radius*2+2)*audio.channels*2)
                val a=PcmReader(file,audio.channels,cacheBytes);val b=PcmReader(other,audio.channels,cacheBytes)
                val stride=(audio.rate/8000).coerceAtLeast(1)
                fun score(offset:Int):Double {
                    if(end+offset-width<0 || end+offset>audio.frames)return -1.0
                    if(cancelled()) throw CancellationException()
                    var sum=0.0;var weakest=1.0
                    val block=width/3
                    repeat(3) {part->
                        var xy=0.0;var xx=0.0;var yy=0.0
                        for(i in part*block until (part+1)*block step stride)for(channel in 0 until audio.channels) {
                            val x=a.sample(start-width+i,channel).toDouble();val y=b.sample(end+offset-width+i,channel).toDouble()
                            xy+=x*y;xx+=x*x;yy+=y*y
                        }
                        val value=xy/sqrt((xx*yy).coerceAtLeast(1.0))
                        sum+=value;weakest=min(weakest,value)
                    }
                    return sum/3*0.65+weakest*0.35
                }
                val baseline=score(0);var best=baseline;var shift=0
                for(offset in -radius..radius step stride) {
                    val value=score(offset)
                    if(value>best){best=value;shift=offset}
                }
                val coarse=shift
                for(offset in max(-radius,coarse-stride)..min(radius,coarse+stride)) {
                    val value=score(offset)
                    if(value>best){best=value;shift=offset}
                }
                return if((best>0.80 && best>baseline+0.025) || (best>0.65 && best>baseline+0.12))end+shift else end
            }
        }
    }
    private fun writeWavHeader(out: OutputStream, size: Int, rate: Int, channels: Int) {
        fun word(value: Int) { out.write(value and 255); out.write((value ushr 8) and 255) }
        fun dword(value: Int) { word(value); word(value ushr 16) }
        out.write("RIFF".toByteArray()); dword(size+36); out.write("WAVEfmt ".toByteArray()); dword(16)
        word(1); word(channels); dword(rate); dword(rate*channels*2); word(channels*2); word(16)
        out.write("data".toByteArray()); dword(size)
    }
}
