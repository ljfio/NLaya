# 6. NuGet packages and CI

**Goal:** make NLaya installable, and keep parity with Python checked automatically.

## Packages

| Package | Depends on | Notes |
|---|---|---|
| `NLaya` | `Microsoft.ML.Tokenizers`, `System.Numerics.Tensors`, `Microsoft.Extensions.Logging.Abstractions` | No native code |
| `NLaya.TorchSharp` | `NLaya`, `TorchSharp` (managed only) | The app adds `TorchSharp-cpu` or `TorchSharp-cuda-linux/-windows`; say so in the package README |
| `NLaya.Onnx` | `NLaya`, `Microsoft.ML.OnnxRuntime` | Mention `Microsoft.ML.OnnxRuntime.Gpu` for CUDA |
| `NLaya.Extensions.AI` | `NLaya`, `Microsoft.Extensions.AI` | From steps 1–2 |

In `Directory.Build.props` (shared) or the csprojs:
- `PackageId`, `Description` (already set per project), `PackageReadmeFile` (the root README, or
  a trimmed one per package), `PackageLicenseExpression` (`Apache-2.0`, already set),
  `RepositoryUrl=https://github.com/ljfio/NLaya`, `PackageTags`, `PackageIcon` (optional).
- `ContinuousIntegrationBuild=true` in CI, plus `IncludeSymbols` with `snupkg`. The .NET 8+ SDK
  includes SourceLink, so nothing else is needed.
- Attribution: say it's a port of `laya` by Convai Innovations / NandhaKishorM (Apache-2.0), and
  that the model weights are theirs.
- Versioning: start at `0.1.0`. Use a tag-driven version (for example MinVer), or keep `Version` in
  `Directory.Build.props` and bump by hand; ask the user.
- Keep test projects and samples `IsPackable=false` (already the case).
- Check the packed nupkg contains the embedded resources (`lang_data.json`, `email_data.json`,
  `presets.json`); they're `EmbeddedResource`, so they're inside the DLL.

## CI (GitHub Actions)

1. **`build.yml`**, on push and PR, on `ubuntu-latest` and `macos-latest`:
   - `actions/setup-dotnet` with the SDK from `global.json`.
   - `dotnet build -warnaserror`, then `dotnet test --project tests/NLaya.Tests`.
   - The tokenizer and sequence cases need the tokenizers: run
     `pip install -U huggingface_hub` and
     `hf download convaiinnovations/laya --include "tokenizer/*"` plus the same for
     `laya-multilingual`, with `actions/cache` on `~/.cache/huggingface` keyed by the laya commit.
     Without them those cases skip; they don't fail.
2. **`parity.yml`**, nightly or on demand (`workflow_dispatch`), `ubuntu-latest`:
   - Cache and `hf download` both repos (~2.5 GB), then run
     `NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity` with TorchSharp CPU.
   - ONNX parity needs the exports: either download them from the Hub repos in
     [step 4b](04-reduce-custom-code.md), or run `export_onnx.py` (Python + torch, slow) and cache
     the result. Set `NLAYA_ONNX_ROOT`.
3. **`release.yml`**, on a `v*` tag: `dotnet pack -c Release`, then `dotnet nuget push` to nuget.org
   with a `NUGET_API_KEY` secret. The user has to create the key and decide on publishing;
   **don't publish without asking**.

## Before the first release

- Decide the repo's visibility (it's private now) and the package names. Check `NLaya` is free
  on nuget.org.
- Run the full parity suite on Linux as well as macOS. So far it has only run on macOS arm64.
- Try to reproduce the one-off exit crash with TorchSharp and ONNX Runtime in one process
  (see [README](README.md) gotchas) on Linux.
