"""Portable regression suite; synthetic audio only. Run: python -m unittest -v.

Set ASTRE8_REFERENCE_CLASSPATH and JAVA_EXE to also compare with the original
Kotlin implementation. These comparisons are test tooling, not runtime needs.
"""
from dataclasses import replace
import io
import importlib.util
import json
import os
from pathlib import Path
import struct
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
import numpy as np
import astre8 as a
import astre8_metadata as metadata

RATE = a.RATE
PITCHES = np.array([93., 123., 164., 219., 292., 389., 519., 692.])
NOTES = np.array([1, 5, 2, 7, 0, 4, 6, 3, 5, 0, 7, 2, 4, 1, 6, 0, 3, 7, 5, 2, 6, 4, 0, 1])


def phrase(notes=NOTES, phase=0.):
    i = np.arange(len(notes) * RATE // 2)
    within, time = (i % (RATE // 2)) / RATE, i / RATE
    frequency = PITCHES[np.asarray(notes)[i // (RATE // 2)]]
    envelope = np.minimum(1., within / .018) * np.minimum(1., (.5 - within) / .035)
    return ((.24 + .055 * np.sin(time * 2.17) + .025 * np.sin(time * 5.3)) * envelope *
            (np.sin(2 * np.pi * frequency * within + phase)
             + .22 * np.sin(2 * np.pi * frequency * .501 * within - phase * .7)
             + .12 * np.sin(2 * np.pi * frequency * 1.37 * within + phase * .4))).astype(np.float32)


def intro():
    return (np.random.default_rng(1297).random(RATE * 3, dtype=np.float32) - .5) * .13


def fixtures():
    body = phrase()
    second = body * .62 * np.minimum(1., (len(body) - np.arange(len(body))) / (RATE * 2))
    changed = phrase((NOTES + 3) % 8)
    changed[:RATE * 2] = body[:RATE * 2]
    changed[-RATE * 2:] = body[-RATE * 2:]
    boundary = phrase([7, 1, 4, 0])
    middle = phrase([2, 6, 3, 5, 1, 7, 2, 4, 6, 0, 5, 3, 7, 6, 1, 2, 0, 4, 3, 6])
    yield "two_pass", np.concatenate([intro(), body, body]), a.LoopKind.REPEATED, 12.
    yield "gain_fade", np.concatenate([intro(), body, second]), a.LoopKind.REPEATED, 12.
    yield "phase", np.concatenate([intro(), body, phrase(phase=1.2)]), a.LoopKind.REPEATED, 12.
    yield "three_phase", np.concatenate([intro(), body, phrase(phase=1.2), phrase(phase=.7)]), a.LoopKind.REPEATED, 12.
    yield "short_restart", np.concatenate([intro(), body, body[:RATE * 3] * .72]), a.LoopKind.REPEATED, 12.
    yield "contradictory_body", np.concatenate([intro(), body, changed]), None, None
    yield "single_pass", np.concatenate([boundary, middle, boundary]), a.LoopKind.INFERRED, 12.
    irregular = np.concatenate([middle[:len(middle) // 2], middle[len(middle) // 2:len(middle) // 2 + 26], middle[len(middle) // 2:]])
    yield "off_grid", np.concatenate([boundary, irregular, boundary]), a.LoopKind.INFERRED, 12.013
    yield "unrelated_notes", phrase([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 4, 6, 1, 3, 5, 7, 1, 4, 7, 2, 5, 0, 3, 6, 2, 6, 3, 7, 4, 0, 5, 1]), None, None
    yield "noise", (np.random.default_rng(74).random(RATE * 12, dtype=np.float32) - .5) * .1, None, None
    yield "silence", np.zeros(RATE * 10, np.float32), None, None


def write_wave(path, samples, rate=8000):
    samples = np.asarray(samples, dtype="<i2")
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def append_chunk(path, kind, payload):
    data = path.read_bytes()
    data += kind + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")
    data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
    path.write_bytes(data)


def smpl(start, end, loop_type=0, count=1, fraction=0):
    return struct.pack("<15I", 0, 0, 0, 60, 0, 0, 0, count, 0, 0, loop_type, start, end - 1, fraction, 0)


def comment_packet(fields):
    encoded = [field.encode() for field in fields]
    return struct.pack("<I", 4) + b"test" + struct.pack("<I", len(fields)) + b"".join(struct.pack("<I", len(field)) + field for field in encoded)


class DetectorTests(unittest.TestCase):
    def test_musical_structure_cases(self):
        for name, pcm, kind, period in fixtures():
            with self.subTest(name=name):
                result = a.detect(pcm)
                if kind is None:
                    self.assertIsNone(result)
                else:
                    self.assertIsNotNone(result)
                    self.assertEqual(kind, result.kind)
                    self.assertAlmostEqual(period, (result.end - result.start) / RATE, delta=.15)
                    self.assertTrue(0 <= result.start < result.end < len(pcm))

    def test_fade_and_candidate_budget(self):
        body = phrase(np.concatenate([NOTES, NOTES[::-1], (NOTES + 2) % 8]))
        messages = []
        result = a.detect(np.concatenate([intro(), body, body]), diagnostics=messages.append)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(36, (result.end - result.start) / RATE, delta=.15)
        candidates = next(line for line in messages if line.startswith("Candidate periods:")).split(":")[1].split(",")
        self.assertLessEqual(len(candidates), 32)

    def test_half_sample_shift_and_fade(self):
        body = phrase()
        second = (body * .5 + np.r_[body[1:], body[-1]] * .5) * .76 * np.minimum(1., (len(body) - np.arange(len(body))) / (RATE * 4))
        result = a.detect(np.concatenate([intro(), body, second]))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(12, (result.end - result.start) / RATE, delta=.15)
        self.assertTrue(result.fading)

    def test_conflict_cannot_fall_back_to_inference(self):
        first = phrase()
        second = phrase((NOTES + 3) % 8)
        second[:RATE * 4] = first[:RATE * 4]
        second[-RATE * 2:] = first[:RATE * 2]
        messages = []
        self.assertIsNone(a.detect(np.r_[intro(), first, second], diagnostics=messages.append))
        self.assertIn("Conflicting repeated forms rule out single-pass inference", messages)

    def test_cancellation_before_and_during(self):
        with self.assertRaises(a.CancelledError):
            a.detect(phrase(), cancelled=lambda: True)
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls == 5

        with self.assertRaises(a.CancelledError):
            a.detect(np.r_[intro(), phrase(), phrase()], cancelled=cancel)
        self.assertEqual(5, calls)

    def test_invalid_and_short_input(self):
        self.assertIsNone(a.detect(np.zeros(2)))
        for samples in (np.array([np.nan]), np.array([np.inf]), np.zeros((2, 2)), np.zeros(RATE * 1200 + 1)):
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                a.detect(samples)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.pcm = self.root / "source.pcm"
        self.samples = np.random.default_rng(92).integers(-32768, 32768, (24000, 2), dtype=np.int16)
        self.samples.tofile(self.pcm)

    def test_decode_stereo_weighting_and_rate(self):
        source = self.root / "in.wav"
        write_wave(source, self.samples, 8000)
        decoded = a.decode_wave(source, self.root / "decoded.pcm")
        self.assertEqual(self.pcm.read_bytes(), decoded.path.read_bytes())
        mono = self.samples[:, 0].astype(np.float32) / 32768 * np.float32(.6) + self.samples[:, 1].astype(np.float32) / 32768 * np.float32(.4)
        np.testing.assert_allclose(decoded.analysis, mono.reshape(-1, 4).mean(axis=1), atol=1e-7)

    def test_decode_mono_nonintegral_rate(self):
        source = self.root / "mono.wav"
        write_wave(source, np.arange(44100, dtype=np.int16), 44100)
        decoded = a.decode_wave(source, self.root / "decoded.pcm")
        self.assertEqual(2000, len(decoded.analysis))
        self.assertEqual(44100, decoded.frames)
        self.assertEqual(1, decoded.channels)

    def test_decode_preserves_existing_working_file(self):
        source = self.root / "in.wav"
        write_wave(source, self.samples)
        original = self.pcm.read_bytes()
        with self.assertRaises(FileExistsError):
            a.decode_wave(source, self.pcm)
        self.assertEqual(original, self.pcm.read_bytes())

    def test_decode_cancellation_cleans_partial_pcm(self):
        source = self.root / "in.wav"
        write_wave(source, self.samples)
        target = self.root / "cancelled.pcm"
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 2

        with self.assertRaises(a.CancelledError):
            a.decode_wave(source, target, cancelled=cancel)
        self.assertFalse(target.exists())

    def test_truncated_input_rejected(self):
        source = self.root / "broken.wav"
        write_wave(source, self.samples)
        source.write_bytes(source.read_bytes()[:-9])
        with self.assertRaises(ValueError):
            a.decode_wave(source, self.root / "out.pcm")
        self.assertFalse((self.root / "out.pcm").exists())

    def test_native_unsupported_format(self):
        source = self.root / "8bit.wav"
        with wave.open(str(source), "wb") as stream:
            stream.setparams((1, 1, 8000, 0, "NONE", "none"))
            stream.writeframes(bytes(800))
        with self.assertRaises(a.UnsupportedAudioError):
            a.decode_wave(source, self.root / "out.pcm")

    def test_missing_ffmpeg_message(self):
        source = self.root / "compressed.bin"
        source.write_bytes(b"not a WAV")
        with patch("astre8.shutil.which", return_value=None), self.assertRaisesRegex(ValueError, "FFmpeg"):
            a.decode_audio(source, self.root / "out.pcm")

    def test_scalar_render_and_arbitrary_byte_seeks(self):
        for channels in (1, 2):
            pcm = self.root / f"{channels}.pcm"
            samples = self.samples[:, :channels]
            samples.tofile(pcm)
            for start, end, overlap in ((1000, 17000, 300), (0, 17000, 300), (1000, 17000, 0)):
                with self.subTest(channels=channels, start=start, overlap=overlap):
                    plan = a.ExtensionPlan(8000, channels, start, end, overlap, len(samples))
                    total, fade_frames = 40000, 8000
                    expected = bytearray()
                    for frame in range(total):
                        source = frame if frame < end else start + (0 if start >= overlap else overlap) + (frame - end) % plan.period
                        incoming = start - (overlap if start >= overlap else 0) + source - (end - overlap) if overlap and source >= end - overlap else -1
                        mix = np.float32(source - (end - overlap)) / np.float32(overlap) if incoming >= 0 else np.float32(0)
                        fade = np.float32(total - frame - 1) / np.float32(fade_frames) if frame >= total - fade_frames else np.float32(1)
                        for channel in range(channels):
                            value = np.float32(samples[source, channel])
                            if incoming >= 0:
                                value = value * (np.float32(1) - mix) + np.float32(samples[incoming, channel]) * mix
                            value = min(32767, max(-32768, int(np.floor(float(value * fade) + .5))))
                            expected.extend(struct.pack("<h", value))
                    with a.ExtensionWaveReader(pcm, plan, 5000, 1) as reader:
                        actual = reader.read(0, reader.length)
                        self.assertEqual(bytes(expected), actual[44:])
                        rng = np.random.default_rng(53)
                        for _ in range(35):
                            pos = int(rng.integers(0, reader.length + 5))
                            count = int(rng.integers(0, 1500))
                            self.assertEqual(actual[pos:pos + count], reader.read(pos, count))
                        self.assertEqual(b"", reader.read(reader.length, 42))
                        self.assertEqual(b"", reader.read(0, 0))

    def test_shifted_fade_and_extrema(self):
        plan = a.ExtensionPlan(8000, 2, 1000, 16000, 0, len(self.samples))
        with a.ExtensionWaveReader(self.pcm, plan, 5000, 1, 4000) as reader:
            data = reader.read(44, reader.length - 44)
            samples = np.frombuffer(data, dtype="<i2").reshape(-1, 2)
            np.testing.assert_array_equal(samples[:1000], self.samples[:1000])
            self.assertFalse(samples[-4000:].any())

    def test_hour_stream_reads_small_ranges(self):
        plan = a.ExtensionPlan(8000, 2, 2000, 17000, 300, 24000)
        with a.ExtensionWaveReader(self.pcm, plan, 3600000, 5) as reader:
            self.assertEqual(44 + 3600 * 8000 * 4, reader.length)
            self.assertEqual(1024, len(reader.read(reader.length - 1024, 4096)))
            self.assertEqual(44, len(reader.read(0, 44)))

    def test_invalid_plans_and_read_bounds(self):
        for kwargs in ({"start": -1}, {"end": 30000}, {"overlap": 8000}, {"channels": 3}, {"rate": 2000}):
            values = dict(rate=8000, channels=2, start=1000, end=16000, overlap=200, source_frames=24000)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                a.ExtensionPlan(**values)
        plan = a.ExtensionPlan(8000, 2, 1000, 16000, 200, 24000)
        for duration in (3000, 0, -1, 999999999):
            with self.assertRaises(ValueError):
                plan.wave_bytes(duration)
        with a.ExtensionWaveReader(self.pcm, plan, 5000) as reader:
            with self.assertRaises(ValueError):
                reader.read(-1, 5)
            with self.assertRaises(ValueError):
                reader.read(0, -1)
        with self.assertRaises(ValueError):
            reader.read(0, 1)
        short = self.root / "short.pcm"
        short.write_bytes(b"\0\0")
        with self.assertRaises(ValueError):
            a.ExtensionWaveReader(short, plan, 5000)

    def test_render_exact_duration_fade_and_no_overwrite(self):
        plan = a.ExtensionPlan(8000, 2, 1000, 16000, 200, 24000)
        output = self.root / "extended.wav"
        a.render(self.pcm, plan, output, 5000, 1)
        with wave.open(str(output), "rb") as reader:
            self.assertEqual(40000, reader.getnframes())
            self.assertEqual(2, reader.getnchannels())
            self.assertEqual(8000, reader.getframerate())
            reader.setpos(39999)
            self.assertEqual(bytes(4), reader.readframes(1))
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            a.render(self.pcm, plan, output, 5000)
        self.assertEqual(original, output.read_bytes())

    def test_render_cancellation_cleans_partial_output(self):
        plan = a.ExtensionPlan(8000, 2, 1000, 16000, 200, 24000)
        output = self.root / "cancelled.wav"
        checks = 0

        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 3

        with self.assertRaises(a.CancelledError):
            a.render(self.pcm, plan, output, 60000, cancelled=cancel)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("*.part")))

    def test_embedded_points_preserve_original_samples(self):
        audio = a.DecodedAudio(self.pcm, 8000, 2, 24000, np.empty(0), a.EmbeddedLoop(123, 17456, 8000, "test"))
        plan = a.plan(audio, a.LoopRegion(0, 1, 1., 1., False, a.LoopKind.EMBEDDED), 30)
        self.assertEqual((123, 17456, 123), (plan.start, plan.end, plan.overlap))
        audio.embedded_loop = a.EmbeddedLoop(0, 17456, 8000, "test")
        plan = a.plan(audio, a.LoopRegion(0, 1, 1., 1., False, a.LoopKind.EMBEDDED))
        self.assertEqual(0, plan.overlap)

    def test_acoustic_plan_coordinates(self):
        audio = a.DecodedAudio(self.pcm, 8000, 2, 24000, np.empty(0))
        plan = a.plan(audio, a.LoopRegion(200, 4000, 1., 1., False, a.LoopKind.MANUAL), 30)
        self.assertEqual((800, 16000, 240), (plan.start, plan.end, plan.overlap))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.path = self.root / "loop.wav"
        write_wave(self.path, np.zeros(24000, np.int16))

    def test_wave_smpl_inclusive_end_and_post_audio_chunk(self):
        append_chunk(self.path, b"smpl", smpl(123, 20000))
        loop = a.read_embedded_loop(self.path, 8000, 24000)
        self.assertEqual((123, 20000, 8000), (loop.start, loop.end, loop.rate))

    def test_wave_rejects_ambiguous_or_invalid_loops(self):
        original = self.path.read_bytes()
        for start, end, kind, count, fraction in ((123, 25000, 0, 1, 0), (123, 120, 0, 1, 0),
                                                  (0, 20000, 1, 1, 0), (0, 20000, 0, 2, 0), (0, 20000, 0, 1, 1)):
            self.path.write_bytes(original)
            append_chunk(self.path, b"smpl", smpl(start, end, kind, count, fraction))
            with self.subTest(start=start, end=end, kind=kind, count=count, fraction=fraction):
                self.assertIsNone(a.read_embedded_loop(self.path, 8000, 24000))

    def test_metadata_rate_mismatch_truncation_and_cancellation(self):
        append_chunk(self.path, b"smpl", smpl(0, 20000))
        self.assertIsNone(a.read_embedded_loop(self.path, 44100, 24000))
        self.path.write_bytes(self.path.read_bytes()[:-5])
        self.assertIsNone(a.read_embedded_loop(self.path, 8000, 24000))
        with self.assertRaises(a.CancelledError):
            a.read_embedded_loop(self.path, 8000, 24000, lambda: True)

    def test_flac_comments_and_conflicting_values(self):
        info = bytearray(34)
        info[10:18] = ((8000 << 44) | 24000).to_bytes(8, "big")
        for fields, valid in ((["LOOP_START=123", "LOOPLENGTH=10000"], True),
                               (["LOOPSTART=123", "LOOPEND=10123", "LOOPLENGTH=9999"], False),
                               (["LOOPSTART=123", "LOOPSTART=124", "LOOPEND=10123"], False),
                               (["LOOPSTART=123", "LOOPEND=10123", "LOOPTYPE=backward"], False),
                               (["LOOPSTART=0:01", "LOOPEND=10123"], False)):
            tags = comment_packet(fields)
            self.path.write_bytes(b"fLaC" + b"\0\0\0\x22" + info + b"\x84" + len(tags).to_bytes(3, "big") + tags)
            loop = a.read_embedded_loop(self.path, 8000, 24000)
            with self.subTest(fields=fields):
                if valid:
                    self.assertEqual((123, 10123), (loop.start, loop.end))
                else:
                    self.assertIsNone(loop)

    def test_vorbis_crc_and_sample_coordinates(self):
        def page(packet, sequence, flags):
            lacing = bytes([255] * (len(packet) // 255) + [len(packet) % 255])
            header = bytearray(b"OggS\0" + bytes([flags]) + bytes(8) + struct.pack("<II", 42, sequence) + bytes(4) + bytes([len(lacing)]))
            crc = metadata._ogg_crc(header + lacing + packet)
            header[22:26] = struct.pack("<I", crc)
            return header + lacing + packet

        ident = b"\x01vorbis" + struct.pack("<I", 0) + b"\x01" + struct.pack("<I", 8000) + bytes(12) + b"\x76\x01"
        tags = b"\x03vorbis" + comment_packet(["LOOPSTART=123", "LOOPEND=10123"]) + b"\x01"
        self.path.write_bytes(page(ident, 0, 2) + page(tags, 1, 0))
        loop = a.read_embedded_loop(self.path, 8000, 24000)
        self.assertEqual((123, 10123), (loop.start, loop.end))
        corrupt = bytearray(self.path.read_bytes())
        corrupt[-2] ^= 1
        self.path.write_bytes(corrupt)
        self.assertIsNone(a.read_embedded_loop(self.path, 8000, 24000))

    def test_metadata_budget_and_unsupported_tags(self):
        for data in (b"ID3" + bytes(100), b"OggS" + bytes(30), b"RIFF" + bytes(20)):
            self.path.write_bytes(data)
            self.assertIsNone(a.read_embedded_loop(self.path, 8000, 24000))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.source = self.root / "input with spaces.wav"
        body = phrase()
        signal = np.repeat(np.r_[intro(), body, body], 4)
        write_wave(self.source, np.rint(signal * 24000).astype(np.int16))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(Path(a.__file__).resolve()), *map(str, args)],
                              capture_output=True, text=True, timeout=120)

    def test_analyze_and_auto_extend(self):
        result = self.run_cli(self.source, "--analyze")
        self.assertEqual(0, result.returncode, result.stderr)
        detected = json.loads(result.stdout)
        self.assertEqual("REPEATED", detected["kind"])
        self.assertAlmostEqual(12., detected["end_seconds"] - detected["start_seconds"], delta=.15)
        output = self.root / "output with spaces.wav"
        result = self.run_cli(self.source, output, "--duration", 31, "--fade", 1)
        self.assertEqual(0, result.returncode, result.stderr)
        with wave.open(str(output)) as stream:
            self.assertEqual(31 * 8000, stream.getnframes())

    def test_manual_no_overwrite_and_invalid_arguments(self):
        output = self.root / "output.wav"
        result = self.run_cli(self.source, output, "--duration", 31, "--start", 3, "--end", 15)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual("MANUAL", json.loads(result.stdout)["kind"])
        result = self.run_cli(self.source, output, "--duration", 31, "--start", 3, "--end", 15)
        self.assertEqual(1, result.returncode)
        for arguments in (("--analyze", "--start", 3), ("--analyze", "--fade", "nan"),
                          ("--analyze", "--crossfade", -1), ("--analyze", "--end", 2)):
            with self.subTest(arguments=arguments):
                self.assertEqual(2, self.run_cli(self.source, *arguments).returncode)

    def test_no_loop_and_missing_input_exit_codes(self):
        write_wave(self.source, np.zeros(80000, dtype=np.int16))
        self.assertEqual(2, self.run_cli(self.source, "--analyze").returncode)
        self.assertEqual(1, self.run_cli(self.root / "missing.wav", "--analyze").returncode)


@unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg"), "Install portable test extras for FFmpeg integration tests")
class CompressedAudioTests(unittest.TestCase):
    def test_flac_ogg_and_mp3_decode_and_render(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), binary)
            binary.chmod(0o755)
            source = root / "input.wav"
            signal = np.repeat(phrase(), 4)
            write_wave(source, np.rint(signal * 24000).astype("<i2"))
            original = source.read_bytes()
            for suffix, codec in (("flac", "flac"), ("ogg", "libvorbis"), ("mp3", "libmp3lame")):
                compressed = root / ("input." + suffix)
                converted = subprocess.run([str(binary), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
                                            "-c:a", codec, "-metadata", "LOOPSTART=1000", "-metadata", "LOOPEND=90000", str(compressed)],
                                           capture_output=True, timeout=30)
                self.assertEqual(0, converted.returncode, converted.stderr)
                with self.subTest(format=suffix), patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ.get("PATH", "")}):
                    decoded = a.decode_audio(compressed, root / (suffix + ".pcm"))
                    self.assertEqual((8000, 2, 96000), (decoded.rate, decoded.channels, decoded.frames))
                    if suffix in ("flac", "ogg"):
                        self.assertIsNotNone(decoded.embedded_loop)
                        self.assertEqual((1000, 90000), (decoded.embedded_loop.start, decoded.embedded_loop.end))
                    else:
                        self.assertIsNone(decoded.embedded_loop)
                    plan = a.plan(decoded, a.LoopRegion(250, 22000, 1., 1., False, a.LoopKind.MANUAL))
                    output = root / (suffix + "-extended.wav")
                    a.render(decoded.path, plan, output, 15000, 1)
                    with wave.open(str(output)) as result:
                        self.assertEqual(120000, result.getnframes())
            self.assertEqual(original, source.read_bytes())

    def test_decoder_failure_cleans_working_files(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), binary)
            binary.chmod(0o755)
            source, pcm = root / "bad.mp3", root / "output.pcm"
            source.write_bytes(b"invalid compressed audio")
            with patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ.get("PATH", "")}), self.assertRaisesRegex(ValueError, "FFmpeg"):
                a.decode_audio(source, pcm)
            self.assertFalse(pcm.exists())


@unittest.skipUnless(os.environ.get("ASTRE8_REFERENCE_CLASSPATH"), "Optional Kotlin reference classpath not configured")
class KotlinParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.command = [os.environ.get("JAVA_EXE", "java"), "-cp", os.environ["ASTRE8_REFERENCE_CLASSPATH"],
                        "io.github.fdwgg.astre8.ReferenceKt"]

    def test_detector_matches_kotlin_on_shared_audio(self):
        for name, pcm, _, _ in fixtures():
            pcm = pcm.astype("<f4")
            fixture = self.root / "analysis.f32"
            pcm.tofile(fixture)
            result = subprocess.run(self.command + ["detect", str(fixture)], capture_output=True, text=True, timeout=120)
            self.assertEqual(0, result.returncode, result.stderr)
            reference = json.loads(result.stdout)
            portable = a.detect(pcm)
            with self.subTest(name=name):
                if reference is None:
                    self.assertIsNone(portable)
                else:
                    self.assertIsNotNone(portable)
                    self.assertEqual(reference["kind"], portable.kind)
                    self.assertAlmostEqual(reference["end"] - reference["start"], portable.end - portable.start, delta=RATE * .15)
                    self.assertAlmostEqual(reference["start"], portable.start, delta=RATE * .25)
                    self.assertEqual(reference["fading"], portable.fading)

    def test_renderer_byte_matches_kotlin(self):
        for channels in (1, 2):
            samples = np.random.default_rng(99).integers(-32768, 32768, (24000, channels), dtype=np.int16)
            pcm = self.root / "source.pcm"
            samples.astype("<i2").tofile(pcm)
            for start, end, overlap, fade, offset in ((1000, 17000, 300, 1, 0), (0, 17000, 300, 1, 0),
                                                      (1000, 17000, 0, 0, 0), (1000, 17000, 300, 1, 4000)):
                output = self.root / "reference.wav"
                result = subprocess.run(self.command + ["render", str(pcm), str(output), "8000", str(channels),
                                         str(start), str(end), str(overlap), "24000", "5000", str(fade), str(offset)],
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(0, result.returncode, result.stderr)
                plan = a.ExtensionPlan(8000, channels, start, end, overlap, 24000)
                with self.subTest(channels=channels, start=start, overlap=overlap, fade=fade, offset=offset), \
                        a.ExtensionWaveReader(pcm, plan, 5000, fade, offset) as reader:
                    self.assertEqual(output.read_bytes(), reader.read(0, reader.length))


if __name__ == "__main__":
    unittest.main()
