# Next steps

Each file here is one proposed piece of work, written to be picked up in a fresh session. Read
this page first: it has the shared context the others assume.

| # | File | What it adds | Status |
|---|---|---|---|
| 1 | Microsoft.Extensions.AI middleware | `NLaya.Extensions.AI`: guardrail, router, tools | Done |
| 2 | Dependency injection | `AddLaya` / `AddKeyedLaya` / `AddLayaRouter`, settings, warm-up | Done |
| 3 | [03-decide-typed-schemas.md](03-decide-typed-schemas.md) | `agent.Decide<T>()`: a C# type in, typed values out (port of `laya.structured`) | Next |
| 4 | [04-reduce-custom-code.md](04-reduce-custom-code.md) | ONNX exports on the Hub, upstream tokenizer gaps (4a, .NET 10 only, is done) | Any time |
| 5 | [05-mlnet-pipeline-stage.md](05-mlnet-pipeline-stage.md) | An ML.NET `IEstimator`/`ITransformer` | Only if needed |
| 6 | NuGet packages and CI | MinVer versions, `build.yml`, `parity.yml`, `release.yml` with trusted publishing | Done; see "Before the first release" |

Steps 1, 2 and 6 are described in the root `README.md` ("Microsoft.Extensions.AI", "Dependency
injection", "Packages and releases"). Their planning docs were removed once done; they're in git history.

## Where the project stands

NLaya is a .NET port of the Python [`laya`](https://github.com/NandhaKishorM/laya) library. It
answers typed questions (`choice`, `score`, `noul`) about a state in one forward pass. Private repo:
https://github.com/ljfio/NLaya.

- **Works:** all three checkpoints (`english`, `multilingual`, `typed-decisions`) on both backends
  (TorchSharp, ONNX Runtime). `LayaAgent.Predict` / `PredictBatch`, `Router` (including `RouteBatch` /
  `PredictBatch`, Python's `route_batch` / `predict_batch`), `LanguageDetector`, `EmailCleaner`,
  `Presets`, hooks and per-language temperatures are all implemented. `PredictStreamAsync` (a .NET
  addition) streams any `ILayaPredictor` a chunk at a time.
- **Verified:** against golden fixtures from the Python library at laya commit `970dc8c`
  (`tools/fixtures/LAYA_COMMIT`). That covers 2,593 unit tests (tokenization, prompts, JSON,
  1,104 routing cases, 259 email cases) and model parity for all three checkpoints on both backends.
- **Also works:** `NLaya.Extensions.AI` (the .NET equivalent of the LangChain integrations
  `LayaGuardrail` and `LayaRouter`, plus `AIFunction` tools and DI registration), NuGet packaging and
  GitHub Actions. Everything targets .NET 10 only.
- **Not ported yet:** `decide` / `laya.structured` (step 3), `shortlist`, `LayaTriage` / `LayaEvaluator`
  as chat middleware (`LayaTools.Triage` covers triage as a tool), remote `base_url` calls, the HTTP
  server, MCP, CLI and training.
- **Layout:** see "Project layout" in the root `README.md`. The library is one type per file.
  `ILayaBackend.Run` only maps a padded batch to logits; tokenization, prompt layout, calibration
  and decoding all live in `src/NLaya`.

## How the user wants this built

- **Thin layer, Microsoft libraries first.** Use `Microsoft.ML.Tokenizers`,
  `System.Numerics.Tensors`, TorchSharp, `Microsoft.ML.OnnxRuntime` and `Microsoft.Extensions.*`
  before writing anything by hand. Only port what laya itself defines, and say why when something
  has to be hand-written.
- **No downloading inside the library.** Models come from `hf download ...` into the Hugging Face
  cache, and `HfCache` resolves them read-only.
- **One type per file.** Rewrite for clarity where it helps.
- **Commit and push to GitHub** when work is done.

## Before the first release

- **nuget.org trusted publishing.** Add a policy on nuget.org (username menu > Trusted Publishing):
  owner `ljfio`, repository `NLaya`, workflow file `release.yml`, no environment. Add the nuget.org
  profile name as the `NUGET_USER` repository secret. Because the repo is private, the policy starts
  "temporarily active" for 7 days and becomes permanent after the first successful publish.
- **Decide visibility.** The repo is private; the package READMEs link to it. `NLaya` was free on
  nuget.org when checked (September 2026).
- **Run parity on Linux.** `parity.yml` does that (TorchSharp only). ONNX parity in CI waits on step 4b
  (exports on the Hub), since exporting in CI means installing torch.
- **macOS runners cost 10x minutes on private repos.** `build.yml` runs on Ubuntu and macOS; drop macOS
  if minutes matter.
- Try to reproduce the one-off exit crash with TorchSharp and ONNX Runtime in one process on Linux.

## Commands

```bash
hf download convaiinnovations/laya                 # english + multilingual/ + typed-decisions/ (Router default)
hf download convaiinnovations/laya-multilingual    # standalone multilingual (Laya.MultilingualModel)

dotnet build                                        # warnings are errors; unused usings are errors (IDE0005)
dotnet test --project tests/NLaya.Tests             # fast; tokenizer cases skip if the models are not cached
NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity
NLAYA_PARITY=1 NLAYA_ONNX_ROOT=$PWD/onnx dotnet test --project tests/NLaya.Parity   # needs onnx/<name>/ exports
dotnet run --project samples/NLaya.Quickstart
dotnet run --project samples/NLaya.ChatGuardrail
dotnet pack NLaya.slnx -c Release -o artifacts       # version from git tags (MinVer); release by pushing a v* tag
```

ONNX exports live in `./onnx/<name>/` (gitignored). Recreate them with laya's script, run from a laya
checkout at the pinned commit:

```bash
git clone https://github.com/NandhaKishorM/laya && git -C laya checkout 970dc8c && cd laya
for n in multilingual:convaiinnovations/laya-multilingual english:convaiinnovations/laya typed_decisions:convaiinnovations/laya-typed-decisions; do
  uv run --python 3.12 --with . --with onnx --with onnxruntime --with onnxscript \
    python laya-ts/scripts/export_onnx.py --repo "${n#*:}" --out-dir "../NLaya/onnx/${n%%:*}"
done
```

## Gotchas already paid for

- **PyPI lags GitHub.** The PyPI release and GitHub `main` both say version 0.3.20, but `main` has
  unreleased changes (for example `mixed_segment` in language detection). Fixtures and embedded
  tables come from the commit in `LAYA_COMMIT`; regenerate with `tools/fixtures/make_fixtures.py`
  (see `tools/fixtures/README.md`).
- **The embedded JSON tables are generated, not hand-edited.** `lang_data.json`, `email_data.json`
  and `presets.json` all come out of the Python source.
- **Tokenizer.** `LayaTokenizer` wraps `Microsoft.ML.Tokenizers.BpeTokenizer`. The wrapper supplies
  what the library lacks for the Gemma-style multilingual vocabulary: added tokens matched
  longest-first (with `lstrip`), a `▁` prefix per segment, and byte fallback. The `laya-ts` port
  differs from HF `tokenizers` here; treat HF as ground truth.
- **`TensorPrimitives.SoftMax` does not subtract the max.** The act head's logits are around 1e3 and
  overflow to NaN, so use `Decoder.StableSoftmax`.
- **Python string semantics.** `PyStr` handles code-point lengths, `repr` and `%.0f` rounding.
  `'İ'.lower()` is two code points in Python. Python's `\w` has no combining marks, so
  `LanguageDetector` and `EmailCleaner` spell out their character classes.
- **Python JSON.** States are serialized like `json.dumps(ensure_ascii=False)` (`PythonJson`). One
  byte of difference changes the tokens.
- **macOS git ignores case.** A `*.onnx` ignore rule once hid `src/NLaya.Onnx/`, so ignore rules
  are now directory-only.
- **zsh doesn't word-split** `${x:+--flag $y}`, so pass separate arguments.
- **xUnit v3 on the Microsoft Testing Platform.** `global.json` sets the test runner, so use
  `dotnet test --project <proj>`.
- **Exit crash, seen once.** A parity run with both libtorch and ONNX Runtime loaded crashed at
  process exit (`recursive_mutex lock failed`, exit code 134). It hasn't been reproduced.
- **The Router uses the bundle repo.** Its default is `convaiinnovations/laya` with subfolders, as
  in Python. The bundle's `multilingual/` is a separate 680 MB copy from the standalone repo.
- **Evicted agents aren't disposed.** When the Router evicts an agent, it drops the reference and
  lets the GC free it, because a concurrent call may still hold it. `Router.Dispose` disposes the
  agents it loaded.
- **Free TorchSharp intermediates per layer.** `LayaNetwork.Encode` gives each encoder layer its own
  `DisposeScope`. With one scope for the whole pass, an 8-row batch of ~640 tokens peaked at ~17 GB,
  which killed the CI runner; now the parity suite peaks at ~5.6 GB. `SafeTensors.Load` uses plain
  arrays, not `ArrayPool.Shared`, which kept the largest (~790 MB) buffers alive after loading.
- **Parity tests hold one checkpoint at a time.** `ParityFixture` swaps agents, and `ByModelOrderer`
  groups test cases by checkpoint so each loads about once.
- **The guard preset trips on ordinary requests.** Python's `LayaGuardrail` applies one threshold (0.5)
  to noul and score answers alike. `harm_severity` has an expected level of about 0.51 for "What time
  does the office open?", so the default blocks it. NLaya keeps the default for parity; the README, the
  sample and the end-to-end test set `Thresholds["harm_severity"] = 2`.
- **`hf download --include` takes one pattern per flag.** Extra bare arguments are read as filenames and
  the `--include` is silently ignored.
- **The solution is `NLaya.slnx`.** Pass it explicitly (`dotnet build NLaya.slnx`) if another solution file appears.
- **The model card's Hindi example** answers `sales` (0.58), not `billing`. Python does the same, so
  it isn't a port bug.
