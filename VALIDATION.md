# Initial release validation

## Portable Python implementation

Validated on 28 October 2026 with Python 3.12.14 and NumPy 2.3.5 on Windows.
The complete portable suite ran **34 tests in 46.833 seconds: all passed, no
failures, errors, or skips**, with both optional integrations enabled.

- Real CLI subprocesses: automatic analysis/extension, manual points, paths with
  spaces, invalid arguments, missing files, no-loop exit status, and no overwrite.
- Acoustic fixtures: complete repeated forms, phase/gain differences, final fade,
  short restart, contradictory body, inferred boundaries, off-grid timing,
  half-sample shift, noise, silence, and bounded candidate selection.
- File handling: original PCM preservation, stereo analysis weighting, noninteger
  downsample ratio, malformed/truncated audio, cancellation, and temporary cleanup.
- Metadata: WAV inclusive-to-exclusive points, FLAC and Vorbis comments, Ogg CRC,
  duplicate/conflicting values, invalid modes, sample-rate/frame bounds, and cancellation.
- Rendering: independent scalar comparison, arbitrary byte seeks, exact length,
  ending silence, shifted fade, mono/stereo, zero-start loops, no crossfade, and
  a one-hour virtual stream without allocating that full output.
- Compressed integration: actual FLAC, Ogg Vorbis, and MP3 encoding/decoding via
  the imageio-ffmpeg test binary; authored FLAC/Ogg points and delayed-codec tag
  exclusion; exact duration of extended decoded audio.
- Kotlin oracle: 11 shared detector fixtures agree on accept/reject and evidence
  type, with period/boundary tolerances; **8 rendered WAV cases match byte for
  byte** against the original Kotlin implementation.
- Python wheel builds successfully and installs into an isolated target folder.

The optional Kotlin checks use `portable/Reference.kt`, compiled with the four
original engine files and Kotlin stdlib. Java is not a portable runtime dependency.
Portable CI is configured for Windows/macOS/Linux and Python 3.10/3.12/3.13.
Only the local Windows execution above is claimed here; CI logs establish the
results on other operating systems after publication.

## Android extraction

Validated on 27 October 2026 with JDK 17, Gradle 8.9, and Android SDK 35.

Command: `gradle testDebugUnitTest lintRelease assembleRelease`.

- Build: successful.
- Regression tests: 52 passed across 7 suites; 0 failures, errors, or skips.
- Release AAR: generated successfully, including the compiled integration example.
- Android lint: 0 errors, 2 warnings. The warnings concern the deliberately
  retained Android Gradle plugin version and the existing free-space check using
  `File.usableSpace`.
- All four extracted engine files match my (currently private) music app's version 8 source
  after replacing only the package declaration.
- Fixtures are synthetic; no personal audio or application data is distributed.

These checks run on the JVM and build the Android library. Physical-device
decoding and audible transition quality have not been retested for this extraction.
