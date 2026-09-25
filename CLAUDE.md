# NLaya

A .NET port of the Python [`laya`](https://github.com/NandhaKishorM/laya) decision models
(`convaiinnovations/laya`, `-multilingual`, `-typed-decisions`). Start with `README.md` for usage
and layout, and `docs/next-steps/README.md` for current status, planned work and known gotchas.

## Principles

- **Thin layer over Microsoft libraries.** Reach for `Microsoft.ML.Tokenizers`,
  `System.Numerics.Tensors`, TorchSharp, `Microsoft.ML.OnnxRuntime` and `Microsoft.Extensions.*`
  before writing code. Hand-port only what laya itself defines, and say why.
- **Parity with Python is the spec.** Behaviour is checked against golden fixtures generated from the
  laya commit in `tools/fixtures/LAYA_COMMIT` (GitHub `main`, which is ahead of PyPI). New ported
  behaviour gets fixtures from `tools/fixtures/make_fixtures.py`, not hand-written expectations.
  Embedded tables (`lang_data.json`, `email_data.json`, `presets.json`) are generated, never edited
  by hand.
- **Capability parity, .NET-shaped API.** Everything Python can do, NLaya can do, but the public API
  uses .NET types (enums such as `Checkpoint`, `TimeSpan`, typed lists, DI registration). Python's
  shapes live at the edges: `ToJson()` / `Parse` / `FromJson` (the dict formats), error and routing-reason
  wording, and the byte-exact model input (`PythonJson`, `PyStr`, `PyValue`, internal).
- **No downloads in the library.** Models come from `hf download` into the HF cache (`HfCache`).
- **One type per file**, file named after the type. Match the surrounding comment style: XML docs
  on public members (enforced for `src/` by CS1591), short "why" comments.

## Conventions

`.editorconfig` is the style guide and the build enforces it: with `TreatWarningsAsErrors`, formatting
(IDE0055), unused usings, private members and parameters, naming, and the `latest-recommended` .NET
analyzers all fail the build. `dotnet format NLaya.slnx` fixes most findings; CI checks
`dotnet format whitespace` and `style` with `--verify-no-changes`. Opt-outs live in `.editorconfig` with a reason (for example CA1308, since laya
lower-cases for parity). Prefer fixing over suppressing; suppress locally with a justification only
when the rule is wrong for that line.

- File-scoped namespaces; one namespace per public area (`NLaya`, `NLaya.Routing`, ...), not per folder.
- `_camelCase` private fields, `s_camelCase` private static mutable fields, PascalCase everything else.
- Libraries `ConfigureAwait(false)` (CA2007); log through `[LoggerMessage]` methods (`LayaLog`); fixed
  regexes use `[GeneratedRegex]`.
- LF line endings (`.gitattributes`), final newline, no trailing whitespace.
- Observability goes through `LayaTelemetry` (`ActivitySource`/`Meter` named "NLaya"), not new logging or hooks.
- Test code used by both test projects (fixture loading, `JsonAssert`, fakes both need) lives in
  `tests/NLaya.Testing`, with the golden fixtures in `tests/NLaya.Testing/Fixtures`.

## Build and test

```bash
dotnet build NLaya.slnx                                 # TreatWarningsAsErrors; unused usings (IDE0005) fail the build
dotnet test --project tests/NLaya.Tests                 # unit + fixture parity (xUnit v3 on MTP: use --project)
NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity # model parity; needs `hf download convaiinnovations/laya` and `.../laya-multilingual`
NLAYA_PARITY=1 NLAYA_ONNX_ROOT=$PWD/onnx dotnet test --project tests/NLaya.Parity   # + ONNX exports in onnx/<name>/
NLAYA_PARITY=1 NLAYA_DEVICE=mps dotnet test --project tests/NLaya.Parity            # on a GPU (mps here, cuda on Linux); NLAYA_DTYPE=bfloat16 too
```

Accuracy and calibration against laya's published numbers: `benchmarks/NLaya.Eval` (see `docs/tuning.md`).

Run both test projects before committing changes to tokenization, prompts, decoding or backends.

## Git

Commit and push to `origin` (private GitHub repo `ljfio/NLaya`) when a piece of work is done.
Releases: pushing a `v*` tag runs `.github/workflows/release.yml`, which publishes to nuget.org
(MinVer takes the version from the tag). Never push a tag without the user asking.
Keep `.gitignore` rules directory-scoped: macOS git ignores case, and a `*.onnx` pattern once hid
`src/NLaya.Onnx/`.
