# Contributing

Bug reports and pull requests are welcome. Include the source format, sample
rate, channel count, expected loop points, actual result, and detector diagnostics
when possible. Share audio only if you have permission to redistribute it.

Build with JDK 17 and Android SDK 35, then run:

```sh
bash ./gradlew testDebugUnitTest lintRelease assembleRelease
```

On Windows use `gradlew.bat` instead of `bash ./gradlew`. Add a synthetic regression
test when changing loop selection, timing, crossfades, metadata interpretation,
or rendering. Keep fixtures generated in tests rather than committing music.
Changes to detector behavior should document their effect on both false matches
and missed loops. Contributions are made under the MIT license.

For the portable Python version:

```sh
python -m pip install './portable[test]'
python -m unittest discover -s portable -v
```

Keep platform-specific behavior behind the audio decoding boundary. Preserve
exclusive-end sample coordinates and add independent audio/output assertions.
When changing shared rules, run the optional Kotlin parity tests documented in
`portable/README.md` as well as the portable suite.
