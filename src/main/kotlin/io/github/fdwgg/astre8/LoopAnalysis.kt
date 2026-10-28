package io.github.fdwgg.astre8

import java.util.concurrent.CancellationException
import kotlin.math.*

enum class LoopKind { REPEATED, INFERRED, MANUAL, EMBEDDED }
data class LoopRegion(
    val start: Long, val end: Long, val confidence: Float, val repeats: Float, val fading: Boolean,
    val kind: LoopKind = LoopKind.REPEATED, val suggestedCrossfadeMs: Int = 0
)

/** Recurrence is a hypothesis until its temporal context and the actual join agree. */
object LoopAnalysis {
    const val RATE = 2000
    private const val HOP = 100 // 50 ms, with overlapping 256 ms FFT windows.
    private const val WINDOW = 512
    private const val BANDS = 12
    private const val MIN_LOOP = RATE * 3
    private data class Features(val shapes: Array<FloatArray>, val fingerprints: Array<FloatArray>, val rms: FloatArray, val tonalFraction: Float)
    private data class Bounds(val first: Int, val last: Int, val fadeStart: Int, val fading: Boolean)
    private data class Recurrence(val region:LoopRegion?=null,val rulesOutInference:Boolean=false)

    fun detect(pcm: FloatArray, strict: Boolean = true, diagnostics: ((String)->Unit)? = null, cancelled: () -> Boolean = { false }): LoopRegion? {
        checkCancelled(cancelled)
        if (pcm.size < RATE * 5) return null
        val features = extract(pcm, cancelled)
        val peak = features.rms.maxOrNull() ?: return null
        if (peak < 0.001f) return null
        val bounds = bounds(features.rms)
        val recurrence=repeated(pcm, features, bounds, strict, diagnostics, cancelled)
        if(recurrence.region!=null)return recurrence.region
        // A lack of repeated evidence can support cautious one-pass inference. Explicit
        // contradictory/ambiguous repeated evidence cannot be erased by that fallback.
        if(recurrence.rulesOutInference){diagnostics?.invoke("Conflicting repeated forms rule out single-pass inference");return null}
        return inferSinglePass(pcm, features, bounds, strict, diagnostics, cancelled)
    }

    private fun extract(pcm: FloatArray, cancelled: () -> Boolean): Features {
        val count = pcm.size / HOP
        val raw = Array(count) { FloatArray(BANDS) }
        val shapes = Array(count) { FloatArray(BANDS) }
        val rms = FloatArray(count)
        val hann = FloatArray(WINDOW) { (0.5 - 0.5 * cos(2 * PI * it / (WINDOW - 1))).toFloat() }
        val real = DoubleArray(WINDOW); val imag = DoubleArray(WINDOW)
        val bandForBin = IntArray(WINDOW / 2) { bin ->
            (ln((bin * RATE.toDouble() / WINDOW).coerceAtLeast(30.0) / 30) / ln(1000.0 / 30) * BANDS).toInt().coerceIn(0, BANDS - 1)
        }
        val binsPerBand = IntArray(BANDS)
        for (bin in 8 until WINDOW / 2) binsPerBand[bandForBin[bin]]++
        var tonal = 0; var active = 0
        for (frame in 0 until count) {
            if (frame % 64 == 0) checkCancelled(cancelled)
            var energy = 0.0
            for (i in 0 until HOP) { val sample = pcm[frame * HOP + i]; energy += sample * sample }
            rms[frame] = sqrt(energy / HOP).toFloat()
            val center = frame * HOP
            for (i in 0 until WINDOW) { real[i] = pcm.getOrElse(center + i - WINDOW / 2) { 0f } * hann[i].toDouble(); imag[i] = 0.0 }
            fft(real, imag)
            var totalPower = 0.0; var peakPower = 0.0
            for (bin in 8 until WINDOW / 2) {
                val power = real[bin] * real[bin] + imag[bin] * imag[bin]
                totalPower += power; peakPower = max(peakPower, power)
                raw[frame][bandForBin[bin]] += power.toFloat()
            }
            if (rms[frame] > 0.003f) { active++; if (peakPower / totalPower.coerceAtLeast(1e-10) > 0.075) tonal++ }
            for (band in 0 until BANDS) raw[frame][band] = ln((raw[frame][band] / binsPerBand[band].coerceAtLeast(1)).coerceAtLeast(1e-8f))
            val mean = raw[frame].average().toFloat()
            for (band in 0 until BANDS) {
                val centered = raw[frame][band] - mean
                raw[frame][band] = centered
                shapes[frame][band] = centered
            }
            normalize(shapes[frame])
        }
        // Whitening removes the persistent instrument/EQ signature: unrelated music or noise
        // must not count as a repeat merely because it has a similar overall spectrum.
        val means = FloatArray(BANDS)
        val deviations = FloatArray(BANDS)
        val threshold = (rms.maxOrNull() ?: 0f) * 0.025f
        var frames = 0
        for (i in raw.indices) if (rms[i] >= threshold) { for (b in 0 until BANDS) means[b] += raw[i][b]; frames++ }
        for (b in 0 until BANDS) means[b] = means[b] / frames.coerceAtLeast(1).toFloat()
        for (i in raw.indices) if (rms[i] >= threshold) for (b in 0 until BANDS) { val d = raw[i][b] - means[b]; deviations[b] += d * d }
        for (b in 0 until BANDS) deviations[b] = sqrt(deviations[b] / frames.coerceAtLeast(1)).coerceAtLeast(0.6f)
        for (i in raw.indices) {
            for (b in 0 until BANDS) raw[i][b] = (raw[i][b] - means[b]) / deviations[b]
            normalize(raw[i])
        }
        return Features(shapes, raw, rms, tonal.toFloat() / active.coerceAtLeast(1))
    }

    private fun repeated(pcm: FloatArray, f: Features, bounds: Bounds, strict: Boolean, diagnostics: ((String)->Unit)?, cancelled: () -> Boolean): Recurrence {
        val minLag = MIN_LOOP / HOP
        // A one-cycle file may include only the beginning of the next cycle. Do not cap the
        // period at 2/3 of the source or demand three independent full-length repetitions.
        val maxLag = bounds.last - bounds.first - RATE / HOP
        if (maxLag < minLag) return Recurrence()
        val candidates = mutableListOf<Pair<Int, Float>>()
        val floor = (f.rms.maxOrNull() ?: 0f) * 0.025f
        for (lag in minLag..maxLag) {
            if (lag % 32 == 0) checkCancelled(cancelled)
            val available = bounds.last - lag - bounds.first
            var sum = 0f; var strong = 0; var tested = 0
            val step = max(1, available / 100)
            for (i in bounds.first until bounds.last - lag step step) {
                if (f.rms[i] < floor || f.rms[i + lag] < floor) continue
                val similarity = pair(f, i, i + lag)
                sum += similarity; if (similarity > 0.72f) strong++; tested++
            }
            if (tested >= 12) candidates.add(lag to (sum / tested * 0.7f + strong.toFloat() / tested * 0.3f))
        }
        // Select actual peaks, then reserve some places for each period scale. Otherwise
        // one broad beat/harmonic peak can use the entire budget before a complete form
        // reaches the expensive waveform checks. Both passes share the same 32-lag cap.
        val byLag=FloatArray(maxLag+3){Float.NEGATIVE_INFINITY}
        for((lag,score)in candidates)byLag[lag]=score
        val peaks = candidates.filter { (lag, score) ->
            (-2..2).all { delta ->
                val other=byLag[lag+delta]
                delta==0 || other<score || (other==score && delta>0)
            }
        }.sortedByDescending { it.second }
        val selected = mutableListOf<Int>()
        fun select(lag:Int) { if(selected.size<32 && selected.none {abs(it-lag)<3})selected.add(lag) }
        val scales = 8
        val logSpan = ln((maxLag + 1).toDouble() / minLag).coerceAtLeast(1e-6)
        fun scale(lag:Int) = (ln(lag.toDouble()/minLag)/logSpan*scales).toInt().coerceIn(0,scales-1)
        val perScale=Array(scales){mutableListOf<Int>()}
        for((lag,_)in peaks)perScale[scale(lag)].let {if(it.size<2)it.add(lag)}
        for(bucket in perScale)for(lag in bucket)select(lag)
        for((lag,_) in peaks)select(lag)
        diagnostics?.invoke("Candidate periods: "+selected.joinToString { "${it*HOP.toFloat()/RATE}s" })
        data class Accepted(val region:LoopRegion,val quality:Float,val precise:Boolean,val coverage:Float,val run:Int)
        val accepted=mutableListOf<Accepted>()
        var rulesOutInference=false
        val required = if (strict) 0.74f else 0.64f
        for (lag in selected) {
            checkCancelled(cancelled)
            val available = bounds.last - lag
            val matchFrames = max(20, min(50, (lag * 0.32f).toInt()))
            if (available - bounds.first < matchFrames) continue
            fun frameScore(i: Int, l: Int): Float {
                if (i + l >= f.rms.size) return -1f
                val r1 = f.rms[i]; val r2 = f.rms[i + l]
                return when {
                    r1 >= floor && r2 >= floor -> pair(f, i, i + l)
                    r1 < floor && r2 < floor -> -0.2f
                    min(r1, r2) >= floor * 0.45f && min(r1, r2) >= max(r1, r2) * 0.35f -> pair(f, i, i + l)
                    else -> -1f
                }
            }
            val scores = FloatArray(available) { i -> frameScore(i, lag) }
            var rolling = 0f; var maximum = -1f; var anchor = -1
            for (i in bounds.first until available) {
                rolling += scores[i]
                if (i - bounds.first >= matchFrames) rolling -= scores[i - matchFrames]
                if (i - bounds.first + 1 >= matchFrames && rolling / matchFrames > maximum) { maximum = rolling / matchFrames; anchor = i - matchFrames + 1 }
            }
            if (maximum < required - 0.04f || anchor < 0) {diagnostics?.invoke("${lag*HOP.toFloat()/RATE}s rejected: spectral=$maximum");continue}
            var left = anchor; var right = anchor + matchFrames
            fun nearby(index: Int):Float {
                val first=max(bounds.first,index-2);val last=min(available-1,index+2)
                var sum=0f;var activeCount=0;var mutualQuiet=0
                for(i in first..last) {
                    if(f.rms[i]<floor && f.rms[i+lag]<floor) mutualQuiet++
                    else { sum+=scores[i]; activeCount++ }
                }
                return if(activeCount>0) sum/activeCount else if(mutualQuiet==last-first+1) required-0.12f else -1f
            }
            // Brief transients/reverb differences or shared rests must not split a genuine two-pass match.
            // A wrong musical section still breaks the run after a sustained mismatch.
            var gap=0
            for(i in anchor-1 downTo bounds.first) {
                if(nearby(i)>required-0.18f){left=i;gap=0}else if(++gap>12)break
            }
            gap=0
            for(i in anchor+matchFrames until available) {
                if(nearby(i)>required-0.18f){right=i+1;gap=0}else if(++gap>12)break
            }
            // If a brief 1-2s drum fill or variation split the passage, merge across the fill
            // when the surrounding passage and overall active agreement remain consistent.
            val bridgeWindow = max(18, min(36, lag / 6))
            val maxFillGap = max(24, min(50, lag / 7))
            if (available - bounds.first >= lag / 2 && left - bounds.first >= bridgeWindow) {
                var earlyRoll = 0f; var earlyStart = -1
                for (i in bounds.first until left) {
                    earlyRoll += scores[i]
                    if (i - bounds.first >= bridgeWindow) earlyRoll -= scores[i - bridgeWindow]
                    if (i - bounds.first + 1 >= bridgeWindow && earlyRoll / bridgeWindow >= required) {
                        val candStart = i - bridgeWindow + 1
                        var endPrev = candStart + bridgeWindow
                        var g = 0
                        for (j in endPrev until left) if (nearby(j) > required - 0.18f) { endPrev = j + 1; g = 0 } else if (++g > 10) break
                        if (left - endPrev <= maxFillGap) { earlyStart = candStart; break }
                    }
                }
                if (earlyStart >= 0) {
                    var activeSum = 0f; var activeStrong = 0; var activeN = 0
                    for (i in earlyStart until right) if (scores[i] >= -0.1f) { activeSum += scores[i]; if (scores[i] > required) activeStrong++; activeN++ }
                    if (activeN > 0 && activeSum / activeN >= required + 0.04f && activeStrong.toFloat() / activeN >= 0.78f) left = earlyStart
                }
            }
            if (available - bounds.first >= lag / 2 && available - right >= bridgeWindow) {
                var lateRoll = 0f; var lateEnd = -1
                for (i in right until available) {
                    lateRoll += scores[i]
                    if (i - right >= bridgeWindow) lateRoll -= scores[i - bridgeWindow]
                    if (i - right + 1 >= bridgeWindow && lateRoll / bridgeWindow >= required) {
                        val candStart = i - bridgeWindow + 1
                        if (candStart - right <= maxFillGap) {
                            lateEnd = i + 1
                            var g = 0
                            for (j in i + 1 until available) if (nearby(j) > required - 0.18f) { lateEnd = j + 1; g = 0 } else if (++g > 10) break
                            break
                        }
                    }
                }
                if (lateEnd > right) {
                    var activeSum = 0f; var activeStrong = 0; var activeN = 0
                    for (i in left until lateEnd) if (scores[i] >= -0.1f) { activeSum += scores[i]; if (scores[i] > required) activeStrong++; activeN++ }
                    if (activeN > 0 && activeSum / activeN >= required + 0.04f && activeStrong.toFloat() / activeN >= 0.78f) right = lateEnd
                }
            }

            val rawAnchor = left * HOP + WINDOW / 2
            val run = (right-left)*HOP-WINDOW
            val length = min(RATE, run/3).coerceAtLeast(RATE/4)
            if (run < length || rawAnchor + lag*HOP + run >= pcm.size) continue
            val anchors = intArrayOf(rawAnchor, rawAnchor+(run-length)/2, rawAnchor+run-length)
            var offset = lag * HOP; var waveform = -1f; var weakestWaveform = -1f; var bestSubSample = 0
            // Search the +/-1 hop neighborhood first; only accept a wider +/-2 hop shift
            // when the waveform itself strongly locks onto an aligned period (>= 0.78f).
            val nearMin = max(MIN_LOOP, lag * HOP - HOP)
            val nearMax = min(pcm.size - rawAnchor - run - 2, lag * HOP + HOP)
            for (candidate in nearMin..nearMax) {
                var sum=0f;var weakest=1f
                for (anchorPoint in anchors) {
                    val value=correlation(pcm,anchorPoint,anchorPoint+candidate,length,2)
                    sum+=value;weakest=min(weakest,value)
                }
                val evidence=sum/anchors.size*0.65f+weakest*0.35f
                if (evidence > waveform) { waveform=evidence;weakestWaveform=weakest;offset=candidate }
            }
            if (waveform < 0.78f) {
                val farMin = max(MIN_LOOP, lag * HOP - 2 * HOP)
                val farMax = min(pcm.size - rawAnchor - run - 2, lag * HOP + 2 * HOP)
                var farWave = waveform; var farWeak = weakestWaveform; var farOffset = offset
                for (candidate in farMin..farMax) {
                    if (candidate in nearMin..nearMax) continue
                    var sum = 0f; var weakest = 1f
                    for (anchorPoint in anchors) {
                        val value = correlation(pcm, anchorPoint, anchorPoint + candidate, length, 2)
                        sum += value; weakest = min(weakest, value)
                    }
                    val evidence = sum / anchors.size * 0.65f + weakest * 0.35f
                    if (evidence > farWave) { farWave = evidence; farWeak = weakest; farOffset = candidate }
                }
                if (farWave >= 0.78f && farWave > waveform + 0.12f) {
                    waveform = farWave; weakestWaveform = farWeak; offset = farOffset
                }
            }
            val coarseOffset = offset
            val refineMin = max(MIN_LOOP, coarseOffset - 1)
            val refineMax = min(pcm.size - rawAnchor - run - 2, coarseOffset + 1)
            for (candidate in refineMin..refineMax) {
                for (sub in -1..1) {
                    var sum = 0f; var weakest = 1f
                    for (anchorPoint in anchors) {
                        val value = alignedCorrelation(pcm, anchorPoint, anchorPoint + candidate, length, 2, sub)
                        sum += value; weakest = min(weakest, value)
                    }
                    val evidence = sum / anchors.size * 0.65f + weakest * 0.35f
                    if (evidence > waveform) { waveform = evidence; weakestWaveform = weakest; offset = candidate; bestSubSample = sub }
                }
            }
            // Only shift the spectral lag when waveform alignment locked onto a genuine
            // sample-accurate period; phase-altered passes must keep their spectral lag.
            val alignedLag = if (waveform >= 0.75f) (offset.toFloat() / HOP).roundToInt().coerceIn(minLag, maxLag) else lag
            if (waveform >= 0.75f) {
                val lagLow = (offset / HOP).coerceIn(minLag, maxLag)
                val lagHigh = ((offset + HOP - 1) / HOP).coerceIn(minLag, maxLag)
                val lagFrac = (offset % HOP).toFloat() / HOP
                if (alignedLag != lag || lagLow != lagHigh) {
                    for (i in bounds.first until available) {
                        val base = max(scores[i], frameScore(i, alignedLag))
                        scores[i] = if (lagLow != lagHigh && lagFrac in 0.2f..0.8f && i + lagHigh < f.rms.size && f.rms[i] >= floor && f.rms[i + lagLow] >= floor && f.rms[i + lagHigh] >= floor) {
                            max(base, pair(f, i, i + lagLow) * (1f - lagFrac) + pair(f, i, i + lagHigh) * lagFrac)
                        } else base
                    }
                }
            }
            // The three alignment anchors locate the lag; they do not establish that
            // the intervening music repeats. Check independent sections across the
            // entire observed passage, with a fixed total sample budget on long files.
            val sectionCount=(run/RATE).coerceIn(3,12)
            val stride=max(2,run/(RATE*8))
            var sectionSum=0f;val sectionScores=FloatArray(sectionCount)
            for(section in 0 until sectionCount) {
                checkCancelled(cancelled)
                val first=rawAnchor+run*section/sectionCount
                val last=rawAnchor+run*(section+1)/sectionCount
                val value=alignedCorrelation(pcm,first,first+offset,last-first,stride,bestSubSample)
                sectionScores[section]=value;sectionSum+=value
            }
            sectionScores.sort()
            val lowerQuarter=sectionScores[(sectionCount-1)/4]
            waveform=min(waveform,sectionSum/sectionCount*0.65f+lowerQuarter*0.35f)
            val lowestSection = if (sectionCount >= 6 && sectionScores[0] >= 0.45f) max(sectionScores[0], sectionScores[1] * 0.96f) else sectionScores.first()
            weakestWaveform=min(weakestWaveform,lowestSection)
            var sum=0f;var strong=0;var count=0
            for(i in left until right)if(scores[i]>=-0.1f){sum+=scores[i];if(scores[i]>required)strong++;count++}
            val passage=sum/count.coerceAtLeast(1)
            val agreement=strong.toFloat()/count.coerceAtLeast(1)
            val variation=variation(f,left,right)

            // A dying fade-out tail at the very end of the recording is not a contradictory section C.
            var observableEnd = available
            if (available - bounds.first >= lag / 2) {
                val studioFadeStart = min(bounds.fadeStart, max(bounds.first + (bounds.last - bounds.first) / 2, bounds.last - 28 * RATE / HOP))
                while (observableEnd > right) {
                    val idx = observableEnd - 1
                    val targetIdx = (idx + alignedLag).coerceAtMost(f.rms.lastIndex)
                    val inFade = bounds.fading && targetIdx >= studioFadeStart && f.rms[targetIdx] < f.rms[idx] * 0.65f
                    val fadedQuiet = f.rms[targetIdx] < max(floor * 1.35f, f.rms[idx] * 0.38f)
                    if (inFade || fadedQuiet) observableEnd-- else break
                }
            }
            val coverage = ((right - left).toFloat() / (observableEnd - bounds.first).coerceAtLeast(1)).coerceAtMost(1f)
            // A local motif must also account for the remaining observable continuation.
            // Otherwise A-B-A-C can be mistaken for A-B repeated, even with perfect A audio.
            val continuation = ((right - left).toFloat() / (observableEnd - left).coerceAtLeast(1)).coerceAtMost(1f)

            // Phase-tolerant matches need changing musical evidence throughout the passage.
            // A persistent chord/instrument spectrum is not proof of a repeated arrangement.
            val preciseMatch=waveform>=0.90f && weakestWaveform>=0.82f
            val spectralMatch=passage>=(if(strict)0.87f else 0.82f) && agreement>=0.78f && variation>=0.06f
            val combinedMatch=run>=offset*0.55f && waveform>=(if(strict)0.72f else 0.62f) && weakestWaveform>=(if(strict)0.58f else 0.48f) &&
                passage>=(if(strict)0.77f else 0.71f) && agreement>=(if(strict)0.68f else 0.58f) && variation>=0.04f
            diagnostics?.invoke("${lag*HOP.toFloat()/RATE}s: spectral=$passage waveform=$waveform agreement=$agreement continuation=$continuation variation=$variation run=${(right-left)*HOP.toFloat()/RATE}s")
            if(!preciseMatch && !spectralMatch && !combinedMatch)continue
            val fullCycleObserved = run >= offset * 0.90f
            val formScalePeriod = offset >= (bounds.last - bounds.first) * HOP * 0.35f
            val requiredContinuation = if (fullCycleObserved && run >= offset * 1.15f) 0.58f
                else if (fullCycleObserved) (if (strict) 0.72f else 0.62f)
                else (if (strict) 0.78f else 0.68f)
            if(continuation<requiredContinuation) {
                if(preciseMatch && formScalePeriod && run>=min(RATE*3f,offset*0.30f))rulesOutInference=true
                diagnostics?.invoke("${lag*HOP.toFloat()/RATE}s rejected: contradictory continuation")
                continue
            }
            val minimumRun=if(preciseMatch)RATE else min(RATE*3,(offset*0.45f).toInt())
            if(run<minimumRun || variation<0.025f)continue
            // A late returning riff must not hide contradictory earlier passages. A
            // fully observed cycle already proves the form and may have a separate
            // outro; a short restart must explain all of its observable continuation.
            var validationStart=left
            if(run<offset*0.85f) {
                val history=max(20,min(40,lag/6))
                var sumBefore=0f
                for(i in bounds.first until left) {
                    sumBefore+=scores[i]
                    if(i-bounds.first>=history)sumBefore-=scores[i-history]
                    if(i-bounds.first+1>=history && sumBefore/history>=(if(strict)0.84f else 0.78f)) {
                        validationStart=i-history+1;break
                    }
                }
            }
            val validationEnd=if(run>=offset*0.85f)right else observableEnd
            val sections=((validationEnd-validationStart)/20).coerceIn(1,12)
            var inconsistent=false
            for(section in 0 until sections) {
                val first=validationStart+(validationEnd-validationStart)*section/sections
                val last=validationStart+(validationEnd-validationStart)*(section+1)/sections
                var sumSection=0f;var activeSection=0
                for(i in first until last)if(scores[i]>=-0.1f){sumSection+=scores[i];activeSection++}
                // Ignore a mostly inaudible fade/silence, but never average an audible
                // wrong phrase away using the many matching sections around it.
                if(activeSection>=max(4,(last-first)/4) && sumSection/activeSection<(if(strict)0.68f else 0.60f)) {
                    inconsistent=true;break
                }
            }
            if(inconsistent){
                if(preciseMatch && formScalePeriod && (validationStart<left || run>=min(RATE*3f,offset*0.30f)))rulesOutInference=true
                diagnostics?.invoke("${lag*HOP.toFloat()/RATE}s rejected: contradictory musical section");continue
            }
            // FFT windows smear the first matching frame into the preceding intro. Move
            // far enough into the verified passage to give the crossfade matching preroll,
            // and search for an unfaded, high-agreement seam if the default point is weak.
            val baseStart = left * HOP + min(RATE*3/4,(right-left)*HOP/2)
            if (baseStart + offset >= pcm.size || offset < MIN_LOOP) continue
            fun validSeam(c: Context) = c.gain >= 0.55f && (c.waveform >= 0.68f || c.spectral >= 0.86f || (c.waveform >= 0.56f && c.spectral >= 0.80f))
            fun seamQuality(c: Context, s: Int): Float {
                val unfaded = if (!bounds.fading || s + offset <= bounds.fadeStart * HOP) 0.08f else 0f
                val early = (1f - (s - baseStart).coerceAtLeast(0).toFloat() / max(RATE * 10, offset)).coerceIn(0f, 1f) * 0.04f
                return c.spectral * 0.40f + c.waveform.coerceAtLeast(0f) * 0.30f + c.gain * 0.20f + c.envelope.coerceAtLeast(0f) * 0.10f + unfaded + early
            }
            var start = baseStart
            var seam = context(pcm, f, baseStart, baseStart + offset, min(RATE / 2, baseStart - bounds.first * HOP), subSample = bestSubSample)
            val baseGood = seam != null && validSeam(seam) && (!bounds.fading || baseStart + offset <= bounds.fadeStart * HOP) &&
                seam.gain >= 0.76f && (seam.waveform >= 0.80f || seam.spectral >= 0.88f)
            if (!baseGood) {
                var bestSeamScore = if (seam != null && validSeam(seam)) seamQuality(seam, baseStart) else Float.NEGATIVE_INFINITY
                val searchEnd = min(right * HOP, pcm.size - 1 - offset)
                val stepSamples = max(HOP * 2, (searchEnd - baseStart) / 48)
                var cand = baseStart + HOP * 2
                while (cand <= searchEnd) {
                    val c = context(pcm, f, cand, cand + offset, min(RATE / 2, cand - bounds.first * HOP), subSample = bestSubSample)
                    if (c != null && validSeam(c)) {
                        val s = seamQuality(c, cand)
                        if (s > bestSeamScore + 0.015f) { bestSeamScore = s; start = cand; seam = c }
                    }
                    cand += stepSamples
                }
            }
            val finalSeam = seam ?: continue
            if (!validSeam(finalSeam)) continue
            val end = start + offset
            if (end >= pcm.size) continue
            val confidence = (passage * 0.7f + waveform.coerceAtLeast(0f) * 0.3f).coerceIn(0f, 1f)
            val repeats = (bounds.last*HOP-start).toFloat()/offset
            // Rank evidence actually present in the recording, without assuming it
            // contains exactly two passes or letting a few exact samples override form.
            val observedForm=(run.toFloat()/offset).coerceAtMost(1f)
            val quality=passage*0.50f+waveform.coerceAtLeast(0f)*0.10f+coverage*0.20f+observedForm*0.20f
            accepted+=Accepted(LoopRegion(start.toLong(),end.toLong(),confidence,repeats,bounds.fading,
                suggestedCrossfadeMs=if(waveform<0.85f)80 else 0),quality,preciseMatch,coverage,run)
        }
        val best=accepted.maxWithOrNull(compareBy<Accepted> {it.quality}.thenByDescending {it.region.end-it.region.start})
            ?:return Recurrence(rulesOutInference=rulesOutInference)
        val near=accepted.filter {best.quality-it.quality<=0.012f}
        val bestPeriod = best.region.end - best.region.start
        fun isSubMultiple(small: Long, large: Long): Boolean {
            if (small >= large) return false
            val multiple = (large.toDouble() / small).roundToInt().coerceAtLeast(2)
            return abs(large - small * multiple) <= max(RATE / 5L, small / 100)
        }
        // Only prefer a shorter candidate in near when it is an actual fundamental period
        // whose observed repetition spans the longer candidate, or when periods are unrelated.
        val fundamentalCandidates = near.filter { cand ->
            val p = cand.region.end - cand.region.start
            p == bestPeriod || (isSubMultiple(p, bestPeriod) && cand.run >= bestPeriod * 0.75f) || !isSubMultiple(p, bestPeriod)
        }
        val winner=(fundamentalCandidates.ifEmpty { near }).minBy {it.region.end-it.region.start}
        val period=winner.region.end-winner.region.start
        fun related(other:Long):Boolean {
            val small=min(period,other);val large=max(period,other)
            val multiple=(large.toDouble()/small).roundToInt().coerceAtLeast(1)
            return abs(large-small*multiple)<=max(RATE/5L,small/100)
        }
        if((!winner.precise || winner.coverage<0.60f) && near.any { !related(it.region.end-it.region.start) }) {
            diagnostics?.invoke("Repeated boundary rejected: competing musical forms")
            return Recurrence(rulesOutInference=true)
        }
        return Recurrence(winner.region)
    }

    private fun inferSinglePass(pcm: FloatArray, f: Features, bounds: Bounds, strict: Boolean, diagnostics:((String)->Unit)?, cancelled: () -> Boolean): LoopRegion? {
        if (f.tonalFraction < 0.18f) return null // Never invent a loop in silence/unrelated broadband noise.
        val length = bounds.last - bounds.first
        val minimumBody = max(MIN_LOOP / HOP, (length * 0.48f).toInt())
        val preferredEnd = if (bounds.fading) bounds.fadeStart else bounds.last - 1
        val endFrom = max(bounds.first + minimumBody, preferredEnd - min(100, length / 4))
        val endTo = min(bounds.last - 1, preferredEnd)
        var bestStart = -1; var bestEnd = -1; var bestScore = -1f; var bestSimilarity = -1f
        val peak = f.rms.maxOrNull() ?: return null
        data class Candidate(val start:Int,val end:Int,val context:Context,val gain:Float,val complete:Float,val late:Float,val score:Float)
        // Keep the waveform search bounded even on a long, harmonically static source,
        // while reserving slots for the opening phrase so mid-song chords cannot crowd it out.
        val candidates=java.util.PriorityQueue<Candidate>(compareBy {it.score})
        val earlyCandidates=java.util.PriorityQueue<Candidate>(compareBy {it.score})
        val earlyLimit = bounds.first + max(80, length / 8)
        val history=if(strict)RATE else RATE*3/4
        for (end in endFrom..endTo) {
            checkCancelled(cancelled)
            val startTo = min(bounds.first + (length * 0.42f).toInt(), end - minimumBody)
            for (start in bounds.first + 2..startTo) {
                if (f.rms[start] < peak * 0.08f || f.rms[end] < peak * 0.08f) continue
                // A single matching terminal note is common in unrelated phrases. Require
                // a longer ordered context, waveform agreement, and compatible dynamics.
                if(start*HOP-bounds.first*HOP<history)continue
                val gain = min(f.rms[start], f.rms[end]) / max(f.rms[start], f.rms[end]).coerceAtLeast(1e-8f)
                if(gain<0.65f)continue
                val evidence=context(pcm,f,start*HOP,end*HOP,history,if(strict)0.88f else 0.83f,waveforms=false) ?: continue
                val similarity=evidence.spectral
                if(evidence.envelope<0.78f || evidence.gain<0.65f)continue
                // Prefer a complete musical body and the late, unfaded end over a tiny motif.
                val complete = (end - start).toFloat() / length
                val late = (end - bounds.first).toFloat() / length
                val score=similarity*0.56f+evidence.envelope*0.16f+gain*0.12f+complete*0.08f+late*0.08f
                val cand = Candidate(start*HOP,end*HOP,evidence,gain,complete,late,score)
                if (start <= earlyLimit) {
                    earlyCandidates.add(cand)
                    if (earlyCandidates.size > 32) earlyCandidates.poll()
                } else {
                    candidates.add(cand)
                    if (candidates.size > 48) candidates.poll()
                }
            }
        }
        val mergedCandidates = (earlyCandidates + candidates).sortedByDescending { it.score }
        for(candidate in mergedCandidates) {
            checkCancelled(cancelled)
            var waveform=-1f;var alignedEnd=candidate.end
            // Both start and end are quantized to the 50 ms feature grid, so their relative
            // offset can span +/-1 full hop. Search every sample shift then refine sub-sample phase.
            for(shift in -HOP..HOP) {
                val end=candidate.end+shift
                if(end>preferredEnd*HOP || end<=candidate.start+MIN_LOOP)continue
                val value=contextWaveform(pcm,candidate.start,end,history)
                if(value>waveform){waveform=value;alignedEnd=end}
            }
            val coarseEnd = alignedEnd
            for(shift in -1..1) {
                val end = coarseEnd + shift
                if(end>preferredEnd*HOP || end<=candidate.start+MIN_LOOP)continue
                for(sub in -1..1 step 2) {
                    val value = contextWaveform(pcm, candidate.start, end, history, sub)
                    if(value>waveform){waveform=value;alignedEnd=end}
                }
            }
            if(waveform<(if(strict)0.72f else 0.62f))continue
            val score=candidate.context.spectral*0.40f+waveform*0.28f+candidate.context.envelope*0.12f+candidate.gain*0.08f+candidate.complete*0.06f+candidate.late*0.06f
            if(score>bestScore){bestScore=score;bestStart=candidate.start;bestEnd=alignedEnd;bestSimilarity=candidate.context.spectral}
        }
        if (bestStart < 0) {diagnostics?.invoke("Single-pass boundary rejected: insufficient matching context");return null}
        val start = bestStart
        val end = min(pcm.size - 1, bestEnd)
        if (end - start < MIN_LOOP) return null
        // This is an estimate of musical boundaries, never evidence of an observed repetition.
        diagnostics?.invoke("Single-pass context verified: ${start.toFloat()/RATE}–${end.toFloat()/RATE}s score=$bestScore")
        return LoopRegion(start.toLong(), end.toLong(), min(bestSimilarity,bestScore).coerceIn(0f, 0.89f), 1f, bounds.fading, LoopKind.INFERRED, 120)
    }

    private data class Context(val spectral:Float,val waveform:Float,val gain:Float,val envelope:Float)
    /** Ordered history ending at both boundaries, including separate checks of each third. */
    private fun context(pcm:FloatArray,f:Features,start:Int,end:Int,length:Int,minSpectral:Float=-1f,waveforms:Boolean=true,subSample:Int=0):Context? {
        if(length<RATE/5 || start<length || end<length || max(start,end)>pcm.size)return null
        val frames = length / HOP
        if (frames < 1) return null
        var spectral=0f;var shape=0f;var count=0
        var xx=0.0;var yy=0.0;var xy=0.0
        val a=(start-length)/HOP;val b=(end-length)/HOP
        for(i in 0 until frames) {
            if (a + i >= f.rms.size || b + i >= f.rms.size) return null
            spectral+=pair(f,a+i,b+i);shape+=dot(f.shapes[a+i],f.shapes[b+i]);count++
            val x=f.rms[a+i].toDouble();val y=f.rms[b+i].toDouble()
            xx+=x*x;yy+=y*y;xy+=x*y
        }
        if(xx<1e-8 || yy<1e-8)return null
        val spectralScore=min(spectral/count,shape/count)
        if(spectralScore<minSpectral)return null
        return Context(spectralScore,if(waveforms)contextWaveform(pcm,start,end,length,subSample) else 0f,sqrt(min(xx,yy)/max(xx,yy)).toFloat(),(xy/sqrt(xx*yy)).toFloat())
    }

    private fun contextWaveform(pcm:FloatArray,start:Int,end:Int,length:Int,subSample:Int=0):Float {
        var waveform=0f;var weakest=1f
        val block=length/3
        for(i in 0..2) {
            val value=alignedCorrelation(pcm,start-length+i*block,end-length+i*block,block,1,subSample)
            waveform+=value;weakest=min(weakest,value)
        }
        return waveform/3*0.65f+weakest*0.35f
    }

    private fun variation(f:Features,first:Int,last:Int):Float {
        var sum=0f;var count=0
        for(i in first until last-10 step 5){sum+=1-dot(f.fingerprints[i],f.fingerprints[i+10]);count++}
        return sum/count.coerceAtLeast(1)
    }

    private fun bounds(rms: FloatArray): Bounds {
        val peak = rms.maxOrNull() ?: 0f
        val first = rms.indexOfFirst { it > peak * 0.018f }.coerceAtLeast(0)
        val last = (rms.indexOfLast { it > peak * 0.018f } + 1).coerceAtLeast(first + 1)
        val active = rms.sliceArray(first until last).sorted()
        val reference = active[active.size / 2].coerceAtLeast(1e-6f)
        val tail = rms.takeLast(min(20, rms.size)).average()
        val fading = tail < reference * 0.45
        var fadeStart = last - 1
        if (fading) {
            while (fadeStart > first + (last - first) / 2 && rms[fadeStart] < reference * 0.72f) fadeStart--
            fadeStart = max(fadeStart, last - 12 * RATE / HOP)
        }
        return Bounds(first, last, fadeStart, fading)
    }

    private fun pair(f: Features, a: Int, b: Int) = dot(f.fingerprints[a], f.fingerprints[b]) * 0.7f + dot(f.shapes[a], f.shapes[b]) * 0.3f
    private fun dot(a: FloatArray, b: FloatArray): Float { var sum = 0f; for (i in a.indices) sum += a[i] * b[i]; return sum }
    private fun normalize(a: FloatArray) { val norm = sqrt(a.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-8f); for (i in a.indices) a[i] /= norm }
    private fun checkCancelled(cancelled: () -> Boolean) { if (cancelled()) throw CancellationException() }
    private fun correlation(a: FloatArray, x: Int, y: Int, length: Int, stride: Int): Float = alignedCorrelation(a, x, y, length, stride, 0)
    private fun alignedCorrelation(a: FloatArray, x: Int, y: Int, length: Int, stride: Int, subSample: Int): Float {
        if (subSample == 0) {
            if (x < 0 || y < 0 || max(x, y) + length > a.size) return -1f
            var xy = 0.0; var xx = 0.0; var yy = 0.0
            for (i in 0 until length step stride) { val p = a[x + i].toDouble(); val q = a[y + i].toDouble(); xy += p * q; xx += p * p; yy += q * q }
            return if (xx < 1e-8 || yy < 1e-8) -1f else (xy / sqrt(xx * yy)).toFloat()
        }
        val shift = if (subSample > 0) 1 else -1
        if (x < 0 || min(y, y + shift) < 0 || max(x, max(y, y + shift)) + length > a.size) return -1f
        var xy = 0.0; var xx = 0.0; var yy = 0.0
        for (i in 0 until length step stride) {
            val p = a[x + i].toDouble()
            val q = (a[y + i].toDouble() + a[y + i + shift].toDouble()) * 0.5
            xy += p * q; xx += p * p; yy += q * q
        }
        return if (xx < 1e-8 || yy < 1e-8) -1f else (xy / sqrt(xx * yy)).toFloat()
    }
    private fun fft(real: DoubleArray, imaginary: DoubleArray) {
        var j = 0
        for (i in 1 until real.size) {
            var bit = real.size shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }; j = j xor bit
            if (i < j) { val r = real[i]; real[i] = real[j]; real[j] = r }
        }
        var size = 2
        while (size <= real.size) {
            val angle = -2 * PI / size; val wr = cos(angle); val wi = sin(angle)
            for (base in real.indices step size) {
                var xr = 1.0; var xi = 0.0
                for (k in 0 until size / 2) {
                    val a = base + k; val b = a + size / 2
                    val r = real[b] * xr - imaginary[b] * xi; val im = real[b] * xi + imaginary[b] * xr
                    real[b] = real[a] - r; imaginary[b] = imaginary[a] - im; real[a] += r; imaginary[a] += im
                    val next = xr * wr - xi * wi; xi = xr * wi + xi * wr; xr = next
                }
            }
            size *= 2
        }
    }
}

class ExtensionTimeline(val start: Long, val end: Long, val overlap: Int) {
    init { require(start >= 0 && end > start && overlap >= 0 && overlap * 2 < end - start) }
    private val hasPreroll = start >= overlap
    val period = end - start - if (hasPreroll) 0 else overlap
    data class Frame(val source: Long, val incoming: Long = -1, val mix: Float = 0f)
    fun sourceFrame(output:Long)=if(output<end)output else start+(if(hasPreroll)0 else overlap)+(output-end)%period
    fun incomingFrame(source:Long)=if(overlap>0 && source>=end-overlap)start-(if(hasPreroll)overlap else 0)+source-(end-overlap) else -1L
    fun mix(source:Long)=if(overlap>0 && source>=end-overlap)(source-(end-overlap)).toFloat()/overlap else 0f
    fun frame(output: Long): Frame {
        val source=sourceFrame(output)
        return Frame(source,incomingFrame(source),mix(source))
    }
}
