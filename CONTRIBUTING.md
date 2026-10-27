# Contributing

Bug reports and pull requests are welcome. Include the source format, sample
rate, channel count, expected loop points, actual result, and detector diagnostics
when possible. Share audio only if you have permission to redistribute it.

Build with JDK 17 and Android SDK 35, then run:

```sh
./gradlew testDebugUnitTest lintRelease assembleRelease
```

On Windows use `gradlew.bat` instead of `./gradlew`. Add a synthetic regression
test when changing loop selection, timing, crossfades, metadata interpretation,
or rendering. Keep fixtures generated in tests rather than committing music.
Changes to detector behavior should document their effect on both false matches
and missed loops. Contributions are made under the MIT license.
