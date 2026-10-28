"""NumPy translation of Muzbox/ASTRE-8's version 8 acoustic detector.

Coordinates are samples at RATE, with exclusive ends. This retains v8's full-form
checks, competing-form rejection and cautious single-pass inference. Vectorized
FFT/reductions can differ slightly from Kotlin Float arithmetic near thresholds.
"""
from dataclasses import dataclass
from enum import Enum
from math import floor, log
import heapq
import numpy as np

RATE, HOP, WINDOW, BANDS, MIN_LOOP = 2000, 100, 512, 12, 6000


class CancelledError(Exception):
    """The caller cancelled an operation."""


class LoopKind(str, Enum):
    REPEATED = "REPEATED"
    INFERRED = "INFERRED"
    MANUAL = "MANUAL"
    EMBEDDED = "EMBEDDED"


@dataclass(frozen=True)
class LoopRegion:
    start: int
    end: int
    confidence: float
    repeats: float
    fading: bool
    kind: LoopKind = LoopKind.REPEATED
    suggested_crossfade_ms: int = 0


def checkpoint(cancelled):
    if cancelled():
        raise CancelledError("Operation cancelled")


def _normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


@dataclass
class _Features:
    shapes: np.ndarray
    fingerprints: np.ndarray
    rms: np.ndarray
    tonal: float

    def pair(self, a, b):
        return (np.sum(self.fingerprints[a] * self.fingerprints[b], axis=-1) * .7
                + np.sum(self.shapes[a] * self.shapes[b], axis=-1) * .3)


def _extract(pcm, cancelled):
    count = len(pcm) // HOP
    rms = np.sqrt(np.mean(pcm[:count * HOP].reshape(count, HOP).astype(np.float64) ** 2, axis=1)).astype(np.float32)
    raw = np.zeros((count, BANDS), np.float32)
    tonal = active = 0
    hann = np.hanning(WINDOW).astype(np.float32)
    bins = np.clip((np.log(np.maximum(np.arange(WINDOW // 2) * RATE / WINDOW, 30) / 30)
                    / log(1000 / 30) * BANDS).astype(int), 0, BANDS - 1)
    padded = np.pad(pcm, (WINDOW // 2, WINDOW))
    for first in range(0, count, 64):
        checkpoint(cancelled)
        indexes = np.arange(first, min(first + 64, count))[:, None] * HOP + np.arange(WINDOW)
        power = abs(np.fft.rfft(padded[indexes] * hann, axis=1)[:, :WINDOW // 2]) ** 2
        total = power[:, 8:].sum(axis=1)
        valid = rms[first:first + len(power)] > .003
        active += int(valid.sum())
        tonal += int((valid & (power[:, 8:].max(axis=1) / np.maximum(total, 1e-10) > .075)).sum())
        for band in range(BANDS):
            selected = np.flatnonzero(bins[8:] == band) + 8
            raw[first:first + len(power), band] = np.log(np.maximum(power[:, selected].sum(axis=1)
                                                                     / max(1, len(selected)), 1e-8))
    raw -= raw.mean(axis=1, keepdims=True)
    shapes = _normalize(raw)
    audible = rms >= max(rms) * .025
    means = raw[audible].mean(axis=0) if audible.any() else np.zeros(BANDS)
    deviations = np.maximum(np.sqrt(np.mean((raw[audible] - means) ** 2, axis=0)), .6) if audible.any() else np.full(BANDS, .6)
    return _Features(shapes, _normalize((raw - means) / deviations), rms, tonal / max(1, active))


@dataclass
class _Bounds:
    first: int
    last: int
    fade_start: int
    fading: bool


def _bounds(rms):
    audible = np.flatnonzero(rms > max(rms) * .018)
    first, last = (int(audible[0]), int(audible[-1]) + 1) if len(audible) else (0, 1)
    ordered = np.sort(rms[first:last])
    reference = max(1e-6, float(ordered[len(ordered) // 2]))
    fading = float(rms[-min(20, len(rms)):].mean()) < reference * .45
    fade_start = last - 1
    if fading:
        while fade_start > first + (last - first) // 2 and rms[fade_start] < reference * .72:
            fade_start -= 1
        fade_start = max(fade_start, last - 12 * RATE // HOP)
    return _Bounds(first, last, fade_start, fading)


def _correlation(pcm, x, y, length, stride=1, sub=0):
    shift = 1 if sub > 0 else -1 if sub < 0 else 0
    if min(x, y, y + shift) < 0 or max(x, y, y + shift) + length > len(pcm):
        return -1.
    a = pcm[x:x + length:stride].astype(np.float64)
    b = pcm[y:y + length:stride].astype(np.float64)
    if sub:
        b = (b + pcm[y + shift:y + shift + length:stride]) * .5
    xx, yy = np.dot(a, a), np.dot(b, b)
    return float(np.dot(a, b) / np.sqrt(xx * yy)) if min(xx, yy) >= 1e-8 else -1.


def _context_wave(pcm, start, end, length, sub=0):
    block = length // 3
    values = [_correlation(pcm, start - length + i * block, end - length + i * block, block, sub=sub) for i in range(3)]
    return sum(values) / 3 * .65 + min(values) * .35


@dataclass
class _Context:
    spectral: float
    waveform: float
    gain: float
    envelope: float


def _context(pcm, f, start, end, length, min_spectral=-1., waveforms=True, sub=0):
    if length < RATE // 5 or min(start, end) < length or max(start, end) > len(pcm):
        return None
    frames = length // HOP
    a, b = (start - length) // HOP, (end - length) // HOP
    if frames < 1 or max(a, b) + frames > len(f.rms):
        return None
    indexes = np.arange(frames)
    x, y = f.rms[a:a + frames].astype(np.float64), f.rms[b:b + frames].astype(np.float64)
    xx, yy = np.dot(x, x), np.dot(y, y)
    spectral = min(float(f.pair(a + indexes, b + indexes).mean()),
                   float(np.sum(f.shapes[a:a + frames] * f.shapes[b:b + frames], axis=1).mean()))
    if min(xx, yy) < 1e-8 or spectral < min_spectral:
        return None
    return _Context(spectral, _context_wave(pcm, start, end, length, sub) if waveforms else 0.,
                    float(np.sqrt(min(xx, yy) / max(xx, yy))), float(np.dot(x, y) / np.sqrt(xx * yy)))


def _variation(f, first, last):
    indexes = np.arange(first, last - 10, 5)
    return float((1 - np.sum(f.fingerprints[indexes] * f.fingerprints[indexes + 10], axis=1)).mean()) if len(indexes) else 0.


def _repeated(pcm, f, b, strict, diagnostics, cancelled):
    min_lag, max_lag = MIN_LOOP // HOP, b.last - b.first - RATE // HOP
    if max_lag < min_lag:
        return None, False
    candidates = []
    noise_floor = float(max(f.rms)) * .025
    for lag in range(min_lag, max_lag + 1):
        if lag % 32 == 0:
            checkpoint(cancelled)
        indexes = np.arange(b.first, b.last - lag, max(1, (b.last - lag - b.first) // 100))
        indexes = indexes[(f.rms[indexes] >= noise_floor) & (f.rms[indexes + lag] >= noise_floor)]
        if len(indexes) >= 12:
            values = f.pair(indexes, indexes + lag)
            candidates.append((lag, float(values.mean() * .7 + (values > .72).mean() * .3)))
    by_lag = np.full(max_lag + 3, -np.inf)
    for lag, score in candidates:
        by_lag[lag] = score
    peaks = sorted(((lag, score) for lag, score in candidates
                    if all(d == 0 or by_lag[lag + d] < score or (by_lag[lag + d] == score and d > 0)
                           for d in range(-2, 3))), key=lambda x: -x[1])
    buckets = [[] for _ in range(8)]
    for lag, _ in peaks:
        scale = min(7, max(0, int(log(lag / min_lag) / max(1e-6, log((max_lag + 1) / min_lag)) * 8)))
        if len(buckets[scale]) < 2:
            buckets[scale].append(lag)
    selected = []
    for lag in [v for bucket in buckets for v in bucket] + [lag for lag, _ in peaks]:
        if len(selected) < 32 and all(abs(v - lag) >= 3 for v in selected):
            selected.append(lag)
    diagnostics("Candidate periods: " + ", ".join(f"{lag * HOP / RATE:g}s" for lag in selected))
    accepted, rules_out_inference = [], False
    required = .74 if strict else .64
    for lag in selected:
        checkpoint(cancelled)
        available = b.last - lag
        match_frames = max(20, min(50, int(lag * .32)))
        if available - b.first < match_frames:
            continue

        def frame_scores(l):
            i = np.arange(available)
            result = np.full(available, -1., np.float32)
            i = i[i + l < len(f.rms)]
            r1, r2 = f.rms[i], f.rms[i + l]
            both = (r1 >= noise_floor) & (r2 >= noise_floor)
            quiet = (r1 < noise_floor) & (r2 < noise_floor)
            compatible = (np.minimum(r1, r2) >= noise_floor * .45) & (np.minimum(r1, r2) >= np.maximum(r1, r2) * .35)
            good = both | (~quiet & compatible)
            result[i[quiet]] = -.2
            result[i[good]] = f.pair(i[good], i[good] + l)
            return result

        scores = frame_scores(lag)
        windows = np.convolve(scores[b.first:available], np.ones(match_frames) / match_frames, mode="valid")
        anchor = b.first + int(np.argmax(windows))
        if max(windows) < required - .04:
            continue
        left, right = anchor, anchor + match_frames

        def nearby(index):
            first, last = max(b.first, index - 2), min(available - 1, index + 2)
            i = np.arange(first, last + 1)
            active = ~((f.rms[i] < noise_floor) & (f.rms[i + lag] < noise_floor))
            return float(scores[i[active]].mean()) if active.any() else required - .12

        gap = 0
        for i in range(anchor - 1, b.first - 1, -1):
            if nearby(i) > required - .18:
                left, gap = i, 0
            else:
                gap += 1
                if gap > 12:
                    break
        gap = 0
        for i in range(anchor + match_frames, available):
            if nearby(i) > required - .18:
                right, gap = i + 1, 0
            else:
                gap += 1
                if gap > 12:
                    break
        bridge, max_gap = max(18, min(36, lag // 6)), max(24, min(50, lag // 7))
        if available - b.first >= lag // 2 and left - b.first >= bridge:
            early_start = -1
            rolling = 0.
            for i in range(b.first, left):
                rolling += scores[i]
                if i - b.first >= bridge:
                    rolling -= scores[i - bridge]
                if i - b.first + 1 >= bridge and rolling / bridge >= required:
                    cand_start = i - bridge + 1
                    end_prev, gap = cand_start + bridge, 0
                    for j in range(end_prev, left):
                        if nearby(j) > required - .18:
                            end_prev, gap = j + 1, 0
                        else:
                            gap += 1
                            if gap > 10:
                                break
                    if left - end_prev <= max_gap:
                        early_start = cand_start
                        break
            if early_start >= 0:
                active = scores[early_start:right]
                active = active[active >= -.1]
                if len(active) and active.mean() >= required + .04 and (active > required).mean() >= .78:
                    left = early_start
        if available - b.first >= lag // 2 and available - right >= bridge:
            late_end, rolling = -1, 0.
            for i in range(right, available):
                rolling += scores[i]
                if i - right >= bridge:
                    rolling -= scores[i - bridge]
                if i - right + 1 >= bridge and rolling / bridge >= required:
                    if i - bridge + 1 - right <= max_gap:
                        late_end, gap = i + 1, 0
                        for j in range(i + 1, available):
                            if nearby(j) > required - .18:
                                late_end, gap = j + 1, 0
                            else:
                                gap += 1
                                if gap > 10:
                                    break
                        break
            if late_end > right:
                active = scores[left:late_end]
                active = active[active >= -.1]
                if len(active) and active.mean() >= required + .04 and (active > required).mean() >= .78:
                    right = late_end
        raw_anchor, run = left * HOP + WINDOW // 2, (right - left) * HOP - WINDOW
        length = max(RATE // 4, min(RATE, run // 3))
        if run < length or raw_anchor + lag * HOP + run >= len(pcm):
            continue
        anchors = [raw_anchor, raw_anchor + (run - length) // 2, raw_anchor + run - length]

        def evidence(offset, sub=0):
            values = [_correlation(pcm, a, a + offset, length, 2, sub) for a in anchors]
            return sum(values) / len(values) * .65 + min(values) * .35, min(values)

        offset, waveform, weakest, best_sub = lag * HOP, -1., -1., 0
        near_min, near_max = max(MIN_LOOP, lag * HOP - HOP), min(len(pcm) - raw_anchor - run - 2, lag * HOP + HOP)
        for candidate in range(near_min, near_max + 1):
            score, weak = evidence(candidate)
            if score > waveform:
                waveform, weakest, offset = score, weak, candidate
        if waveform < .78:
            far_wave, far_weak, far_offset = waveform, weakest, offset
            for candidate in range(max(MIN_LOOP, lag * HOP - 2 * HOP), min(len(pcm) - raw_anchor - run - 2, lag * HOP + 2 * HOP) + 1):
                if near_min <= candidate <= near_max:
                    continue
                score, weak = evidence(candidate)
                if score > far_wave:
                    far_wave, far_weak, far_offset = score, weak, candidate
            if far_wave >= .78 and far_wave > waveform + .12:
                waveform, weakest, offset = far_wave, far_weak, far_offset
        coarse = offset
        for candidate in range(max(MIN_LOOP, coarse - 1), min(len(pcm) - raw_anchor - run - 2, coarse + 1) + 1):
            for sub in (-1, 0, 1):
                score, weak = evidence(candidate, sub)
                if score > waveform:
                    waveform, weakest, offset, best_sub = score, weak, candidate, sub
        aligned_lag = min(max_lag, max(min_lag, floor(offset / HOP + .5))) if waveform >= .75 else lag
        if waveform >= .75:
            low, high, fraction = min(max_lag, max(min_lag, offset // HOP)), min(max_lag, max(min_lag, (offset + HOP - 1) // HOP)), offset % HOP / HOP
            if aligned_lag != lag or low != high:
                scores = np.maximum(scores, frame_scores(aligned_lag))
                if low != high and .2 <= fraction <= .8:
                    i = np.arange(available)
                    i = i[i + high < len(f.rms)]
                    i = i[(f.rms[i] >= noise_floor) & (f.rms[i + low] >= noise_floor) & (f.rms[i + high] >= noise_floor)]
                    scores[i] = np.maximum(scores[i], f.pair(i, i + low) * (1 - fraction) + f.pair(i, i + high) * fraction)
        sections = min(12, max(3, run // RATE))
        section_scores = []
        for section in range(sections):
            checkpoint(cancelled)
            first, last = raw_anchor + run * section // sections, raw_anchor + run * (section + 1) // sections
            section_scores.append(_correlation(pcm, first, first + offset, last - first, max(2, run // (RATE * 8)), best_sub))
        section_scores.sort()
        waveform = min(waveform, sum(section_scores) / sections * .65 + section_scores[(sections - 1) // 4] * .35)
        lowest = max(section_scores[0], section_scores[1] * .96) if sections >= 6 and section_scores[0] >= .45 else section_scores[0]
        weakest = min(weakest, lowest)
        active = scores[left:right]
        active = active[active >= -.1]
        passage, agreement = (float(active.mean()), float((active > required).mean())) if len(active) else (0., 0.)
        variation = _variation(f, left, right)
        observable_end = available
        if available - b.first >= lag // 2:
            studio_fade = min(b.fade_start, max(b.first + (b.last - b.first) // 2, b.last - 28 * RATE // HOP))
            while observable_end > right:
                idx = observable_end - 1
                target = min(len(f.rms) - 1, idx + aligned_lag)
                if ((b.fading and target >= studio_fade and f.rms[target] < f.rms[idx] * .65)
                        or f.rms[target] < max(noise_floor * 1.35, f.rms[idx] * .38)):
                    observable_end -= 1
                else:
                    break
        coverage = min(1., (right - left) / max(1, observable_end - b.first))
        continuation = min(1., (right - left) / max(1, observable_end - left))
        precise = waveform >= .90 and weakest >= .82
        spectral = passage >= (.87 if strict else .82) and agreement >= .78 and variation >= .06
        combined = (run >= offset * .55 and waveform >= (.72 if strict else .62) and weakest >= (.58 if strict else .48)
                    and passage >= (.77 if strict else .71) and agreement >= (.68 if strict else .58) and variation >= .04)
        diagnostics(f"{lag * HOP / RATE:g}s: spectral={passage:.5f} waveform={waveform:.5f} agreement={agreement:.5f} continuation={continuation:.5f} variation={variation:.5f}")
        if not (precise or spectral or combined):
            continue
        full_cycle, form_scale = run >= offset * .9, offset >= (b.last - b.first) * HOP * .35
        required_continuation = .58 if full_cycle and run >= offset * 1.15 else (.72 if strict else .62) if full_cycle else (.78 if strict else .68)
        if continuation < required_continuation:
            rules_out_inference |= precise and form_scale and run >= min(RATE * 3, offset * .30)
            continue
        if run < (RATE if precise else min(RATE * 3, int(offset * .45))) or variation < .025:
            continue
        validation_start = left
        if run < offset * .85:
            history, rolling = max(20, min(40, lag // 6)), 0.
            for i in range(b.first, left):
                rolling += scores[i]
                if i - b.first >= history:
                    rolling -= scores[i - history]
                if i - b.first + 1 >= history and rolling / history >= (.84 if strict else .78):
                    validation_start = i - history + 1
                    break
        validation_end = right if run >= offset * .85 else observable_end
        sections = min(12, max(1, (validation_end - validation_start) // 20))
        inconsistent = False
        for section in range(sections):
            first = validation_start + (validation_end - validation_start) * section // sections
            last = validation_start + (validation_end - validation_start) * (section + 1) // sections
            values = scores[first:last]
            values = values[values >= -.1]
            if len(values) >= max(4, (last - first) // 4) and values.mean() < (.68 if strict else .60):
                inconsistent = True
                break
        if inconsistent:
            rules_out_inference |= precise and form_scale and (validation_start < left or run >= min(RATE * 3, offset * .30))
            continue
        base_start = left * HOP + min(RATE * 3 // 4, (right - left) * HOP // 2)
        if base_start + offset >= len(pcm) or offset < MIN_LOOP:
            continue

        def valid(c):
            return c is not None and c.gain >= .55 and (c.waveform >= .68 or c.spectral >= .86 or (c.waveform >= .56 and c.spectral >= .80))

        def quality(c, start):
            unfaded = .08 if not b.fading or start + offset <= b.fade_start * HOP else 0.
            early = min(1., max(0., 1 - max(0, start - base_start) / max(RATE * 10, offset))) * .04
            return c.spectral * .4 + max(0., c.waveform) * .3 + c.gain * .2 + max(0., c.envelope) * .1 + unfaded + early

        start = base_start
        seam = _context(pcm, f, start, start + offset, min(RATE // 2, start - b.first * HOP), sub=best_sub)
        base_good = valid(seam) and (not b.fading or start + offset <= b.fade_start * HOP) and seam.gain >= .76 and (seam.waveform >= .8 or seam.spectral >= .88)
        if not base_good:
            best_quality = quality(seam, base_start) if valid(seam) else -np.inf
            search_end = min(right * HOP, len(pcm) - 1 - offset)
            for cand in range(base_start + HOP * 2, search_end + 1, max(HOP * 2, (search_end - base_start) // 48)):
                c = _context(pcm, f, cand, cand + offset, min(RATE // 2, cand - b.first * HOP), sub=best_sub)
                if valid(c) and quality(c, cand) > best_quality + .015:
                    best_quality, start, seam = quality(c, cand), cand, c
        if not valid(seam) or start + offset >= len(pcm):
            continue
        region = LoopRegion(start, start + offset, min(1., max(0., passage * .7 + max(0., waveform) * .3)),
                            (b.last * HOP - start) / offset, b.fading, suggested_crossfade_ms=80 if waveform < .85 else 0)
        q = passage * .5 + max(0., waveform) * .1 + coverage * .2 + min(1., run / offset) * .2
        accepted.append((region, q, precise, coverage, run))
    if not accepted:
        return None, rules_out_inference
    best = max(accepted, key=lambda a: (a[1], -(a[0].end - a[0].start)))
    near = [a for a in accepted if best[1] - a[1] <= .012]
    best_period = best[0].end - best[0].start

    def submultiple(small, large):
        return small < large and abs(large - small * max(2, floor(large / small + .5))) <= max(RATE // 5, small // 100)

    fundamental = [a for a in near if (a[0].end - a[0].start == best_period
                   or (submultiple(a[0].end - a[0].start, best_period) and a[4] >= best_period * .75)
                   or not submultiple(a[0].end - a[0].start, best_period))]
    winner = min(fundamental or near, key=lambda a: a[0].end - a[0].start)
    period = winner[0].end - winner[0].start

    def related(other):
        small, large = min(period, other), max(period, other)
        return abs(large - small * max(1, floor(large / small + .5))) <= max(RATE // 5, small // 100)

    if (not winner[2] or winner[3] < .6) and any(not related(a[0].end - a[0].start) for a in near):
        diagnostics("Repeated boundary rejected: competing musical forms")
        return None, True
    return winner[0], False


def _infer(pcm, f, b, strict, diagnostics, cancelled):
    if f.tonal < .18:
        return None
    length = b.last - b.first
    minimum_body = max(MIN_LOOP // HOP, int(length * .48))
    preferred_end = b.fade_start if b.fading else b.last - 1
    end_from = max(b.first + minimum_body, preferred_end - min(100, length // 4))
    end_to = min(b.last - 1, preferred_end)
    peak, early_limit = max(f.rms), b.first + max(80, length // 8)
    history = RATE if strict else RATE * 3 // 4
    early, candidates, serial = [], [], 0
    for end in range(end_from, end_to + 1):
        checkpoint(cancelled)
        for start in range(b.first + 2, min(b.first + int(length * .42), end - minimum_body) + 1):
            if min(f.rms[start], f.rms[end]) < peak * .08 or (start - b.first) * HOP < history:
                continue
            gain = min(f.rms[start], f.rms[end]) / max(1e-8, max(f.rms[start], f.rms[end]))
            if gain < .65:
                continue
            c = _context(pcm, f, start * HOP, end * HOP, history, .88 if strict else .83, False)
            if c is None or c.envelope < .78 or c.gain < .65:
                continue
            complete, late = (end - start) / length, (end - b.first) / length
            score = c.spectral * .56 + c.envelope * .16 + gain * .12 + complete * .08 + late * .08
            bucket = early if start <= early_limit else candidates
            heapq.heappush(bucket, (score, serial, start * HOP, end * HOP, c, gain, complete, late))
            serial += 1
            if len(bucket) > (32 if bucket is early else 48):
                heapq.heappop(bucket)
    best, best_score = None, -1.
    for _, _, start, end, c, gain, complete, late in sorted(early + candidates, key=lambda a: -a[0]):
        checkpoint(cancelled)
        waveform, aligned_end = -1., end
        for shift in range(-HOP, HOP + 1):
            candidate_end = end + shift
            if candidate_end > preferred_end * HOP or candidate_end <= start + MIN_LOOP:
                continue
            value = _context_wave(pcm, start, candidate_end, history)
            if value > waveform:
                waveform, aligned_end = value, candidate_end
        coarse = aligned_end
        for shift in (-1, 0, 1):
            candidate_end = coarse + shift
            if candidate_end > preferred_end * HOP or candidate_end <= start + MIN_LOOP:
                continue
            for sub in (-1, 1):
                value = _context_wave(pcm, start, candidate_end, history, sub)
                if value > waveform:
                    waveform, aligned_end = value, candidate_end
        if waveform < (.72 if strict else .62):
            continue
        score = c.spectral * .4 + waveform * .28 + c.envelope * .12 + gain * .08 + complete * .06 + late * .06
        if score > best_score:
            best_score = score
            best = LoopRegion(start, min(len(pcm) - 1, aligned_end), min(.89, max(0., min(c.spectral, score))),
                              1., b.fading, LoopKind.INFERRED, 120)
    if best is not None and best.end - best.start >= MIN_LOOP:
        diagnostics(f"Single-pass context verified: {best.start / RATE:g}–{best.end / RATE:g}s score={best_score:.5f}")
        return best
    diagnostics("Single-pass boundary rejected: insufficient matching context")
    return None


def detect(pcm, strict=True, diagnostics=None, cancelled=lambda: False):
    """Detect a loop in a finite mono signal sampled at exactly 2,000 Hz.

    Returns None when the evidence is insufficient. Raises CancelledError when
    cancelled, ValueError for invalid input. Analysis is bounded to 20 minutes.
    """
    checkpoint(cancelled)
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim != 1 or len(pcm) > RATE * 20 * 60 or not np.isfinite(pcm).all():
        raise ValueError("Expected finite, mono 2 kHz analysis audio, at most 20 minutes")
    if len(pcm) < RATE * 5:
        return None
    log_message = diagnostics or (lambda message: None)
    f = _extract(pcm, cancelled)
    if max(f.rms) < .001:
        return None
    b = _bounds(f.rms)
    region, conflict = _repeated(pcm, f, b, strict, log_message, cancelled)
    if region is not None:
        return region
    if conflict:
        log_message("Conflicting repeated forms rule out single-pass inference")
        return None
    return _infer(pcm, f, b, strict, log_message, cancelled)
