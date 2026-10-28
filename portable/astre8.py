"""ASTRE-8 portable library and CLI. Python 3.10+, NumPy; no Android runtime."""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
import numpy as np
from astre8_detector import RATE, CancelledError, LoopKind, LoopRegion, checkpoint, detect
from astre8_metadata import EmbeddedLoop, read_embedded_loop

VERSION = 8
MAX_SOURCE_BYTES = 512 * 1024 * 1024
MAX_WAVE_BYTES = 2**31 - 1


class UnsupportedAudioError(ValueError):
    """The native decoder needs a conversion step for this audio format."""


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class ExtensionPlan:
    rate: int
    channels: int
    start: int
    end: int
    overlap: int
    source_frames: int
    confidence: float = 1.
    repeats: float = 1.
    kind: LoopKind = LoopKind.MANUAL

    def __post_init__(self):
        for name in ("rate", "channels", "start", "end", "overlap", "source_frames"):
            _integer(getattr(self, name), name)
        if not (8000 <= self.rate <= 384000 and self.channels in (1, 2)
                and 0 <= self.start < self.end <= self.source_frames
                and self.overlap * 2 < self.end - self.start):
            raise ValueError("Invalid PCM format or loop coordinates")

    @property
    def period(self):
        return self.end - self.start - (0 if self.start >= self.overlap else self.overlap)

    def wave_bytes(self, target_ms):
        _integer(target_ms, "target_ms", 1)
        frames = target_ms * self.rate // 1000
        size = frames * self.channels * 2
        if frames <= self.source_frames:
            raise ValueError("Choose a duration longer than the original track")
        if size >= MAX_WAVE_BYTES - 44:
            raise ValueError("Selected duration exceeds the 2 GiB WAV limit")
        return 44 + size


@dataclass
class DecodedAudio:
    path: Path
    rate: int
    channels: int
    frames: int
    analysis: np.ndarray
    embedded_loop: EmbeddedLoop | None = None


def _header(plan, frames):
    size = frames * plan.channels * 2
    return struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", size + 36, b"WAVE", b"fmt ", 16, 1,
                       plan.channels, plan.rate, plan.rate * plan.channels * 2,
                       plan.channels * 2, 16, b"data", size)


class ExtensionWaveReader:
    """Finite, byte-seekable virtual WAV. Work and allocations are bounded per read.

    The caller owns the raw little-endian PCM16 file. Each instance is intended
    for one reader thread. Use it as a context manager to close its memory map.
    """
    def __init__(self, path, plan, target_ms, fade_seconds=5, fade_offset_frames=0):
        if not math.isfinite(fade_seconds) or not 0 <= fade_seconds <= 30:
            raise ValueError("fade_seconds must be between 0 and 30")
        _integer(fade_offset_frames, "fade_offset_frames")
        self.plan = plan
        self.length = plan.wave_bytes(target_ms)
        self.frames = target_ms * plan.rate // 1000
        if fade_offset_frames >= self.frames:
            raise ValueError("Fade offset is outside the extension timeline")
        self.fade_frames = min(int(fade_seconds * plan.rate), self.frames // 4)
        self.fade_offset = fade_offset_frames
        self.fade_start = self.frames - self.fade_frames - fade_offset_frames
        if Path(path).stat().st_size < plan.end * plan.channels * 2:
            raise ValueError("Cached source audio is incomplete")
        self._pcm = np.memmap(path, dtype="<i2", mode="r", shape=(plan.end, plan.channels))
        self._header = _header(plan, self.frames)
        self.closed = False

    def _frames(self, first, count):
        p = self.plan
        output = np.arange(first, first + count, dtype=np.int64)
        preroll = p.start >= p.overlap
        source = np.where(output < p.end, output,
                          p.start + (0 if preroll else p.overlap) + (output - p.end) % p.period)
        incoming = np.where((p.overlap > 0) & (source >= p.end - p.overlap),
                            p.start - (p.overlap if preroll else 0) + source - (p.end - p.overlap), -1)
        values = self._pcm[source].astype(np.float32)
        crossfade = incoming >= 0
        if crossfade.any():
            mix = ((source[crossfade] - (p.end - p.overlap)).astype(np.float32) / p.overlap)[:, None]
            values[crossfade] = values[crossfade] * (np.float32(1) - mix) + self._pcm[incoming[crossfade]].astype(np.float32) * mix
        if self.fade_frames:
            fade = output >= self.fade_start
            gain = (np.maximum(0, self.frames - output[fade] - self.fade_offset - 1).astype(np.float32) / self.fade_frames)[:, None]
            values[fade] *= gain
        # Kotlin roundToInt rounds half toward positive infinity. Keep the addition
        # in float64 so 32766.999f cannot round up during the rounding operation.
        return np.clip(np.floor(values.astype(np.float64) + .5), -32768, 32767).astype("<i2").tobytes()

    def read(self, position, length):
        """Read bytes at an absolute WAV offset; b'' means EOF or zero length."""
        if self.closed:
            raise ValueError("Reader is closed")
        _integer(position, "position")
        _integer(length, "length")
        if not length or position >= self.length:
            return b""
        remaining = min(length, self.length - position)
        result = bytearray()
        if position < 44:
            count = min(remaining, 44 - position)
            result.extend(self._header[position:position + count])
            position += count
            remaining -= count
        frame_bytes = self.plan.channels * 2
        while remaining:
            first, within = divmod(position - 44, frame_bytes)
            count = min(65536, (within + remaining + frame_bytes - 1) // frame_bytes)
            data = self._frames(first, count)
            take = min(remaining, len(data) - within)
            result.extend(data[within:within + take])
            position += take
            remaining -= take
        return bytes(result)

    def close(self):
        if not self.closed:
            self._pcm._mmap.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def decode_wave(source, pcm_path, cancelled=lambda: False, progress=lambda message: None):
    """Decode PCM16 mono/stereo WAV, using bounded blocks, into caller-owned PCM.

    Other formats can first be converted with FFmpeg, or use decode_audio below.
    """
    checkpoint(cancelled)
    source, pcm_path = Path(source), Path(pcm_path)
    if source.resolve() == pcm_path.resolve():
        raise ValueError("PCM working file must differ from the input")
    created = False
    try:
        with wave.open(str(source), "rb") as reader:
            rate, channels, frames = reader.getframerate(), reader.getnchannels(), reader.getnframes()
            if (reader.getsampwidth() != 2 or channels not in (1, 2) or not 8000 <= rate <= 384000
                    or reader.getcomptype() != "NONE"):
                raise UnsupportedAudioError("Use mono/stereo PCM16 WAV at 8–384 kHz, or decode_audio with FFmpeg")
            if frames <= 0 or frames > rate * 20 * 60 or frames * channels * 2 > MAX_SOURCE_BYTES:
                raise ValueError("Source exceeds the 20 minute / 512 MiB analysis limit, or is empty")
            # Match v8's 0.6L + 0.4R downmix and sample-bin averaging. Do not load
            # the source PCM into RAM; retain only the 2 kHz analysis signal.
            sums = np.zeros(frames * RATE // rate + 1, np.float64)
            counts = np.zeros(len(sums), np.int64)
            cursor = 0
            with pcm_path.open("xb") as out:
                created = True
                while cursor < frames:
                    checkpoint(cancelled)
                    block = reader.readframes(min(65536, frames - cursor))
                    if not block or len(block) % (channels * 2):
                        raise ValueError("Truncated WAV audio")
                    values = np.frombuffer(block, dtype="<i2").reshape(-1, channels)
                    mono = values[:, 0].astype(np.float32) / 32768
                    if channels == 2:
                        mono = mono * np.float32(.6) + values[:, 1].astype(np.float32) / 32768 * np.float32(.4)
                    indexes = (np.arange(cursor, cursor + len(values), dtype=np.int64) * RATE) // rate
                    np.add.at(sums, indexes, mono)
                    np.add.at(counts, indexes, 1)
                    out.write(block)
                    cursor += len(values)
                    progress(f"Decoding · {cursor / rate:.1f}s")
            complete = frames * RATE // rate
            analysis = (sums[:complete] / np.maximum(1, counts[:complete])).astype(np.float32)
            return DecodedAudio(pcm_path, rate, channels, frames, analysis,
                                read_embedded_loop(source, rate, frames, cancelled))
    except BaseException:
        # Only remove the working file if this invocation created it.
        # open('xb') never overwrites an existing caller file.
        if created:
            pcm_path.unlink(missing_ok=True)
        raise


def decode_audio(source, pcm_path, cancelled=lambda: False, progress=lambda message: None):
    """WAV natively; FLAC/Ogg/MP3/etc via an optional, separately installed FFmpeg.

    FFmpeg is invoked with an argument list and shell=False. Input is a local
    file. Its network protocols are disabled; output is a temporary WAV.
    """
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError("Source must be a local audio file")
    try:
        return decode_wave(source, pcm_path, cancelled, progress)
    except (wave.Error, EOFError, UnsupportedAudioError):
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("This format needs FFmpeg on PATH; PCM16 WAV works without FFmpeg")
    with tempfile.TemporaryDirectory(prefix="astre8-decode-") as directory:
        decoded = Path(directory) / "decoded.wav"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe",
                   "-i", str(source), "-map", "0:a:0", "-vn", "-t", "1201", "-ac", "2", "-c:a", "pcm_s16le", str(decoded)]
        # Drain stderr into a file to avoid pipe deadlocks on damaged input.
        with (Path(directory) / "decoder.log").open("w+b") as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                while True:
                    checkpoint(cancelled)
                    if decoded.exists() and decoded.stat().st_size > MAX_SOURCE_BYTES + 4096:
                        raise ValueError("Source exceeds the 512 MiB decoding limit")
                    try:
                        code = process.wait(timeout=.1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if code:
                    errors.seek(0)
                    raise ValueError("FFmpeg could not decode this source: " + errors.read(2048).decode("utf-8", "replace").strip())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        audio = decode_wave(decoded, pcm_path, cancelled, progress)
        # Frame coordinates from delay-sensitive MP3/Opus tags remain unsupported.
        audio.embedded_loop = read_embedded_loop(source, audio.rate, audio.frames, cancelled)
        return audio


def plan(audio, region, crossfade_ms=30, cancelled=lambda: False):
    """Build an original-rate plan. Acoustic coordinates use 2 kHz samples."""
    if not math.isfinite(crossfade_ms) or not 0 <= crossfade_ms <= 1000:
        raise ValueError("crossfade_ms must be between 0 and 1000")
    checkpoint(cancelled)
    authored = audio.embedded_loop if region.kind == LoopKind.EMBEDDED else None
    if region.kind == LoopKind.EMBEDDED and (authored is None or authored.rate != audio.rate):
        raise ValueError("Source loop points are unavailable or use a different rate")
    start = authored.start if authored else region.start * audio.rate // RATE
    end = authored.end if authored else region.end * audio.rate // RATE
    if not 0 <= start < end <= audio.frames:
        raise ValueError("Loop points are outside the source")
    if authored is None and region.confidence < 1 and region.kind != LoopKind.MANUAL:
        width = min(audio.rate * 12 // 100, start)
        if width >= audio.rate // 50:
            source = np.memmap(audio.path, mode="r", dtype="<i2", shape=(audio.frames, audio.channels))
            try:
                radius = max(1, audio.rate // (50 if region.kind == LoopKind.INFERRED or region.suggested_crossfade_ms > 0 else 500))
                block, stride = width // 3, max(1, audio.rate // 8000)

                def score(offset):
                    checkpoint(cancelled)
                    if end + offset - width < 0 or end + offset > audio.frames:
                        return -1.
                    values = []
                    for part in range(3):
                        indexes = np.arange(part * block, (part + 1) * block, stride)
                        x = source[start - width + indexes].astype(np.float64).ravel()
                        y = source[end + offset - width + indexes].astype(np.float64).ravel()
                        values.append(float(np.dot(x, y) / np.sqrt(max(1., np.dot(x, x) * np.dot(y, y)))))
                    return sum(values) / 3 * .65 + min(values) * .35

                baseline = best = score(0)
                shift = 0
                for offset in range(-radius, radius + 1, stride):
                    value = score(offset)
                    if value > best:
                        best, shift = value, offset
                coarse = shift
                for offset in range(max(-radius, coarse - stride), min(radius, coarse + stride) + 1):
                    value = score(offset)
                    if value > best:
                        best, shift = value, offset
                if (best > .8 and best > baseline + .025) or (best > .65 and best > baseline + .12):
                    end += shift
            finally:
                source._mmap.close()
    overlap = min(int(max(crossfade_ms, region.suggested_crossfade_ms) * audio.rate / 1000), (end - start) // 8)
    if authored:
        overlap = min(overlap, start)
    return ExtensionPlan(audio.rate, audio.channels, start, end, overlap, audio.frames,
                         region.confidence, region.repeats, region.kind)


def render(pcm_path, extension_plan, output, target_ms, fade_seconds=5, cancelled=lambda: False,
           progress=lambda message: None):
    """Atomically publish a new WAV, preserving existing output and cleaning failures."""
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    checkpoint(cancelled)
    expected = extension_plan.wave_bytes(target_ms)
    if shutil.disk_usage(output.parent).free < expected + 16 * 1024 * 1024:
        raise ValueError("Not enough free disk space")
    descriptor, temporary = tempfile.mkstemp(prefix=".astre8-", suffix=".part", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream, ExtensionWaveReader(pcm_path, extension_plan, target_ms, fade_seconds) as reader:
            cursor = 0
            while cursor < reader.length:
                checkpoint(cancelled)
                data = reader.read(cursor, min(256 * 1024, reader.length - cursor))
                stream.write(data)
                cursor += len(data)
                progress(f"Rendering · {cursor * 100 // reader.length}%")
        # Hard-link publication refuses an existing destination even if another
        # process creates it during rendering. Fall back to exclusive copy for
        # filesystems without hard links, removing only our own incomplete file.
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise
        except OSError:
            created = False
            try:
                with output.open("xb") as dest, open(temporary, "rb") as src:
                    created = True
                    shutil.copyfileobj(src, dest, 256 * 1024)
            except BaseException:
                if created:
                    output.unlink(missing_ok=True)
                raise
        return output
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="ASTRE-8: offline portable soundtrack extender")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--duration", type=float, help="Target duration in seconds, longer than the source")
    parser.add_argument("--start", type=float, help="Manual loop start in seconds (use with --end)")
    parser.add_argument("--end", type=float, help="Manual exclusive loop end in seconds")
    parser.add_argument("--fade", type=float, default=5., help="Ending fade in seconds (0–30)")
    parser.add_argument("--crossfade", type=float, default=30., help="Join crossfade in milliseconds (0–1000)")
    parser.add_argument("--relaxed", action="store_true", help="Relax precision while retaining musical form checks")
    parser.add_argument("--analyze", action="store_true", help="Print detected loop as JSON without rendering")
    parser.add_argument("--ignore-embedded", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be supplied together")
    if not args.analyze and (args.output is None or args.duration is None):
        parser.error("output and --duration are required unless --analyze is used")
    numeric = [v for v in (args.duration, args.start, args.end, args.fade, args.crossfade) if v is not None]
    if not all(math.isfinite(v) and v >= 0 for v in numeric) or args.fade > 30 or args.crossfade > 1000:
        parser.error("Time values must be finite and nonnegative; fade <= 30s and crossfade <= 1000ms")
    try:
        with tempfile.TemporaryDirectory(prefix="astre8-") as directory:
            audio = decode_audio(args.input, Path(directory) / "source.pcm")
            if args.start is not None:
                # Manual points retain original-rate precision by using the same
                # exact-coordinate path as file-authored loops.
                audio.embedded_loop = EmbeddedLoop(round(args.start * audio.rate), round(args.end * audio.rate), audio.rate, "manual")
                region = LoopRegion(0, 1, 1., 1., False, LoopKind.EMBEDDED)
            elif audio.embedded_loop is not None and not args.ignore_embedded:
                region = LoopRegion(0, 1, 1., 1., False, LoopKind.EMBEDDED)
            else:
                region = detect(audio.analysis, not args.relaxed,
                                diagnostics=(lambda m: print(m, file=sys.stderr)) if args.diagnostics else None)
            if region is None:
                print("No suitable loop detected. Supply --start and --end to choose a loop.", file=sys.stderr)
                return 2
            extension_plan = plan(audio, region, args.crossfade)
            summary = {**asdict(extension_plan), "start_seconds": extension_plan.start / audio.rate,
                       "end_seconds": extension_plan.end / audio.rate, "version": VERSION}
            if args.start is not None:
                summary["kind"] = LoopKind.MANUAL
            if not args.analyze:
                render(audio.path, extension_plan, args.output, round(args.duration * 1000), args.fade)
                summary["output"] = str(args.output)
            print(json.dumps(summary, indent=2))
            return 0
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except (OSError, ValueError, wave.Error, EOFError, CancelledError) as error:
        print(f"ASTRE-8: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
