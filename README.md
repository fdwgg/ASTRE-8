# ASTRE-8

**Automatic SoundTrack Extender 8** — an offline Kotlin/Android library that
extends music by finding and repeating a suitable musical passage.

Extracted from the version 8 extender in Muzbox. ASTRE-8 analyzes existing
audio; it does not generate new music or call an AI service. This repository
contains the engine and its tests, rather than a standalone player application.

## What it does

- Detects repeated musical structure and validates the full candidate passage.
- Supports cautious single-pass inference and explicitly supplied loop points.
- Reads authored loops from WAV `smpl`, FLAC tags, and Ogg Vorbis comments.
- Refines joins against the original PCM, with configurable crossfades and an
  ending fade.
- Exports a finite PCM16 WAV, or renders a seekable virtual WAV on demand with
  bounded source buffers.
- Supports cancellation and progress callbacks during decoding and rendering.

## Build

Requirements: JDK 17, Android SDK platform 35, and accepted Android SDK licenses.
Set `ANDROID_HOME` to your SDK installation, or create a local `local.properties`
file containing `sdk.dir=/path/to/android-sdk` (use forward slashes on Windows).

```sh
git clone https://github.com/fdwgg/ASTRE-8.git
cd ASTRE-8
./gradlew testDebugUnitTest lintRelease assembleRelease
```

On Windows run `gradlew.bat testDebugUnitTest lintRelease assembleRelease`.
The wrapper downloads Gradle 8.9 and the build resolves its dependencies on the
first run. The library is written to `build/outputs/aar/ASTRE-8-release.aar`.
Minimum Android version: **8.0 (API 26)**.

## Integrate

Copy the release AAR into your Android application's `libs` folder and add:

```kotlin
dependencies {
    implementation(files("libs/ASTRE-8-release.aar"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib:2.0.20")
}
```

Alternatively include this repository as an Android library module. No Maven
Central publication is required for either approach.

Run decoding, analysis, and rendering on a worker thread. Supply a readable
Android content URI and retain any required URI permission. A compiled example,
`Astre8Example.extendToWave`, is included under `src/main/kotlin`:

```kotlin
import io.github.fdwgg.astre8.Astre8Example

val extended = Astre8Example.extendToWave(
    context = context,
    source = sourceUri,
    output = outputFile, // a new file inside an existing writable directory
    targetMs = 10 * 60 * 1000L,
    cancelled = { Thread.currentThread().isInterrupted },
    progress = { message -> /* post to your UI */ }
)
```

The example decodes to a temporary PCM file, detects a loop, renders the output,
and deletes the temporary PCM in `finally`. It reports an error if no suitable
loop exists. It uses acoustic detection; to honor authored loop metadata, read
`EmbeddedLoopMetadata` from the original file and attach it to `DecodedAudio`,
then supply a `LoopRegion` with `kind = LoopKind.EMBEDDED` when planning.

For streaming, retain the decoded PCM and create a plan with
`AudioExtender.plan(audio, region, crossfadeMs, cancelled)`. Open
`ExtensionWaveReader(audio.file, plan, targetMs, fadeSeconds)` and call
`read(position, buffer, offset, length)` to read arbitrary **byte offsets** in
the virtual WAV, including its 44-byte header. Close the reader when finished.
This avoids writing the complete extended WAV; the source PCM still uses disk.

## API and coordinates

| API | Input / purpose |
| --- | --- |
| `AudioExtender.decode` | Android URI → interleaved little-endian PCM16 file and 2 kHz analysis signal |
| `LoopAnalysis.detect` | Analysis signal sampled at `LoopAnalysis.RATE` (2,000 Hz) → `LoopRegion?` |
| `LoopRegion.start/end` | Analysis-sample coordinates; end is exclusive |
| `EmbeddedLoop.startFrame/endFrame` | Original decoded-frame coordinates; end is exclusive |
| `AudioExtender.plan` | Converts a region to a source-rate `ExtensionPlan` |
| `AudioExtender.render` | Renders a complete extended WAV to disk |
| `ExtensionWaveReader` | Seekable finite WAV rendered as bytes are requested |

`AudioExtender.VERSION` remains **8**. The package is `io.github.fdwgg.astre8`.
Engine classes retain their existing names for a small, auditable extraction.

## Limits

- Detection is heuristic and may return `null`. Repetitive music and soundtracks
  are the intended use case; natural joins are not guaranteed for every song.
- Decoder analysis is limited to 20 minutes / 512 MiB of source PCM. Supported
  input codecs depend on the Android device. Output is mono/stereo PCM16, not a
  lossless copy of a compressed or high-resolution source.
- Target duration must exceed the source duration. A rendered or virtual WAV is
  limited to under 2 GiB. Full export requires enough free disk space.
- MP3 and Opus authored loop tags are intentionally unsupported because decoder
  delay makes sample-coordinate interpretation ambiguous.
- Source files and temporary files belong to the caller. Use separate working
  files for concurrent operations and do not delete PCM while a reader uses it.
- The tests use synthetic audio and local JVM execution. They do not establish
  physical-device codec compatibility or replace listening checks.

## Tests

The extracted regression suites cover complete musical forms, false matches,
single-pass inference, cancellation, metadata bounds, sample-accurate authored
loops, seam refinement, arbitrary seeks, output length, and byte-for-byte
comparison of streaming rendering with a reference renderer. GitHub Actions
builds, tests, and lints the library.

## License

[MIT](LICENSE). See [NOTICE.md](NOTICE.md) for provenance and third-party notices.
The software license does not grant rights to redistribute music processed by it.
