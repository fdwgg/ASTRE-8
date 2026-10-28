# ASTRE-8 portable (Python)

Python 3.10+ and NumPy. Runs on Windows, macOS, and Linux without Android,
Java, a GPU, a network service, or a model download. It includes a command-line
program and a Python API. The engine is MIT licensed like the Android version.

From the repository root:

```sh
python -m pip install ./portable
python -m astre8 input.wav output.wav --duration 600
```

`--duration` is the **total output length in seconds**, longer than the original.
The portable command natively reads mono/stereo PCM16 WAV at 8–384 kHz.
For FLAC, Ogg, MP3 and other formats, install FFmpeg separately and make `ffmpeg`
available on PATH. FFmpeg's license and installation are separate from ASTRE-8.
All input paths must refer to local files. Processing stays on the machine.

```sh
python -m astre8 input.wav --analyze --diagnostics
python -m astre8 input.wav output.wav --duration 600 --fade 5 --crossfade 30
python -m astre8 input.wav output.wav --duration 600 --start 12.5 --end 92.5
```

`--start` and `--end` are manual loop points in seconds; end is exclusive.
`--crossfade` is in milliseconds. `--fade` is in seconds. Valid file-authored
WAV/FLAC/Vorbis loop points take priority over acoustic detection unless
`--ignore-embedded` is used. MP3/Opus authored points remain unsupported because
their delay/pre-skip coordinate interpretation is ambiguous.

Exit codes: 0 success, 1 I/O/format/render failure, 2 insufficient loop evidence
or invalid command arguments, 130 interruption. Existing output files are
preserved. Diagnostics go to stderr; results are JSON on stdout.

## Python API

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from astre8 import decode_audio, detect, plan, render

with TemporaryDirectory() as work:
    audio = decode_audio("input.wav", Path(work) / "source.pcm")
    region = detect(audio.analysis)
    if region is None:
        raise ValueError("No suitable loop")
    recipe = plan(audio, region, crossfade_ms=30)
    render(audio.path, recipe, "extended.wav", target_ms=600_000)
```

This example uses acoustic detection. To use authored metadata, create
`LoopRegion(0, 1, 1., 1., False, LoopKind.EMBEDDED)` when
`audio.embedded_loop` is available. The placeholder analysis coordinates are
ignored in that case; exact decoded-frame coordinates are used.

`detect` requires a **mono 2,000 Hz** analysis signal. Its `LoopRegion` uses
analysis-sample coordinates. `ExtensionPlan` and `EmbeddedLoop` use original
decoded-frame coordinates. `decode_audio` performs the downmix/downsampling.
Detection returns `None` instead of silently selecting an unreliable loop.

For on-demand playback, open
`ExtensionWaveReader(audio.path, recipe, target_ms=600_000, fade_seconds=5)` as a
context manager, then call `read(byte_position, byte_count)`. The virtual file
includes a 44-byte WAV header and supports arbitrary byte offsets. `b""` means
EOF. Memory mapping retains source PCM on disk, with bounded rendering blocks;
the returned bytes still occupy the amount of memory requested by the caller.

Decoding uses up to 512 MiB of temporary PCM and 20 minutes of source audio;
full WAV output is under 2 GiB. Compressed decoding temporarily holds both a
decoded WAV and the raw PCM. The caller owns API working files and must keep
them available while a reader is open. CLI temporary files are cleaned up.
Cancellation predicates raise `CancelledError`; Ctrl+C interrupts the CLI.

## Tests

Run the portable tests from the repository root:

```sh
python -m pip install './portable[test]'
python -m unittest discover -s portable -v
```

The test extra supplies a test-only FFmpeg binary for real FLAC/Ogg/MP3 decode
tests. It is not required for WAV use and is not part of the runtime package.
Without it, only the compressed-format integration tests are skipped.

Tests cover musical forms and false matches, inference, half-sample shifts,
fades, cancellation, WAV/FLAC/Vorbis metadata, malformed files, bounded streaming,
arbitrary byte seeks, independent scalar rendering, exact duration, cleanup,
preserved input/output files, and real CLI subprocesses.

`Reference.kt` is an optional test oracle compiled against the original four
Kotlin engine files. Set `ASTRE8_REFERENCE_CLASSPATH` to those compiled classes
plus Kotlin stdlib, and `JAVA_EXE` to a Java 17+ executable, to enable two
additional tests. They compare detection on identical generated float audio
and byte-for-byte WAV rendering for mono/stereo, crossfades, zero-start loops,
and shifted ending fades. Java is only needed for these comparison tests.

The NumPy detector translates the version 8 rules; it does not promise bitwise
floating-point parity at every threshold. Rendering comparisons are exact for
the tested PCM fixtures. Listening quality and untested audio can vary.
