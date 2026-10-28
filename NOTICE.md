# Source and third-party notices

ASTRE-8 extracts the version 8 song extender from my personal music app. The initial extraction
keeps the four engine source files and seven regression test files unchanged
except for their package declaration (`io.github.fdwgg.astre8`). The new project
build, documentation, and integration example are specific to this release.

The portable Python implementation translates those version 8 detection,
planning, metadata, and rendering rules. NumPy supplies numerical operations
under its BSD-3-Clause license and is installed separately. Floating-point
reductions can differ from Kotlin near detector thresholds. Optional FFmpeg is
a separately installed program under its own build-dependent license. The
imageio-ffmpeg test extra is test tooling and does not ship inside the library.

The engine and project documentation are available under the root MIT license.
No songs, album artwork, trained models, application databases, signing keys,
or original application binaries are included.

The Gradle wrapper scripts and wrapper JAR are from Gradle 8.9 and retain their
own Apache-2.0 licensing. See `licenses/Gradle-LICENSE.txt` and
`licenses/Gradle-NOTICE.txt`. Those files include Gradle's distribution notices;
the complete Gradle distribution is downloaded by the wrapper, not bundled here.

Build dependencies are downloaded from their upstream repositories. Kotlin and
the Android Gradle plugin use Apache-2.0; JUnit 4 uses EPL-1.0. Android SDK
components are separately installed under the Android SDK terms. These
dependencies are not relicensed by the project's MIT license.
