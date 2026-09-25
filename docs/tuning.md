# Tuning NLaya: accuracy, calibration and speed

How to get the most out of the Laya checkpoints from .NET: which checkpoint to use, how to measure
it, how to calibrate and fine-tune it to reach laya's published results (including the comparison
with TypeSafe Jev), and how to make it fast on a GPU. The NLaya commands were run on Apple
silicon (CPU and MPS) while writing this page, and the results are in [Measured with NLaya](#measured-with-nlaya).
The CUDA commands and laya's fine-tuning notebook were not run here.

## What the published numbers are

laya's README and `BENCHMARKS.md` (at the commit in `tools/fixtures/LAYA_COMMIT`) compare Laya with
**TypeSafe Jev 1.13.0**, a closed decision API. laya's authors never ran Jev ("no TypeSafe API access");
its numbers are third-party published.

| Benchmark | Laya | Jev | Laya checkpoint and setup |
|---|---|---|---|
| typed-decisions (400 cases, 2,000 decisions) | **0.766** | 0.727 | `laya-typed-decisions` (fine-tuned), refitted temperatures |
| AG News (4 labels) | **0.950** | 0.910 | in laya's training mix |
| DAIR Emotion (6 labels) | **0.595** | 0.480 | zero-shot |
| Banking77 (77 labels) | 0.425 | **0.870** | 77 options at once; laya's own advice is to stay under ~20 |
| ECE after refitting temperatures (lower is better) | **0.081** | 0.144 | one temperature per (type, option-count bucket) |
| Latency p50, one question | **32.8 ms** | 236–276 ms | `laya-multilingual`, Tesla T4, float32 |

Three things matter for reproducing them:

1. **NLaya gives Python's answers.** The parity suites check tokenization, prompts, logits and
   results against the Python library to 1e-3, so the same checkpoint, questions and temperatures
   give the same accuracy in .NET. Nothing in NLaya's inference needs tuning to match these numbers.
2. **The typed-decisions win comes from fine-tuning.** The base checkpoints score 0.362 (`laya`) and
   0.352 (`laya-multilingual`) on typed-decisions, below its 0.461 majority-class baseline. All of
   the gain comes from `laya-typed-decisions`, fine-tuned on that dataset's training split.
3. **Calibration numbers depend on temperatures, not on the model.** Temperature changes how confident
   an answer is, never which option wins, so it moves ECE, Brier and NLL but not accuracy. The
   published ECE comes from temperatures refitted on held-out data.

## Pick the checkpoint for the job

| Job | Checkpoint | Load it with |
|---|---|---|
| English classification, routing, guardrails | `laya` (ModernBERT-large) | `Laya.LoadAsync("convaiinnovations/laya", ...)` |
| Anything not English, or long documents (to 8,192 tokens) | `laya-multilingual` (mmBERT-base, 2–3× faster) | `Laya.MultilingualModel`, or `Subfolder = "multilingual"` in the bundle |
| The four typed-decisions workflows (customer service, invoices, security incidents, agent traces) | `laya-typed-decisions` | `Subfolder = "typed-decisions"`, or `RouteOptions.Checkpoint = Checkpoint.TypedDecisions` |
| Mixed traffic | `Router` | routes by script and language; `AutoTaskDetection` sends the typed-decisions question sets to that checkpoint |

The English checkpoint doesn't degrade gracefully outside English: on MASSIVE it collapses and stays
confident (Khmer: 0.000 accuracy at 0.952 confidence). Route before you predict.

## Measure first

The evaluation harness runs laya's benchmark suites through NLaya and reports the metric block laya's
harness uses: accuracy, macro-F1, ECE (15 bins, on max p), Brier, NLL, and latency.

```bash
# 1. Export the suites, built exactly as laya's benchmark notebook builds them (same data, seed, wording)
uv run --python 3.12 --with datasets benchmarks/export_suites.py --out benchmarks/suites \
  --only typed_decisions,en.ag_news,en.emotion,en.sst5,xnli.de,massive_intent.de
#    (en.banking77_full needs --with "datasets<4": it is a loading-script dataset)

# 2. Evaluate a checkpoint (Hub id from the HF cache, or a local directory)
dotnet run -c Release --project benchmarks/NLaya.Eval -- \
  --model convaiinnovations/laya --subfolder typed-decisions --device cuda \
  --suite benchmarks/suites/typed_decisions.jsonl --refit per-type --latency 50 --report td.json
```

`benchmarks/suites/` is gitignored, so export the suites locally. Options: `--device cpu|cuda|mps`,
`--dtype float32|bfloat16|float16`, `--onnx <dir>` for an ONNX export, `--batch-size`, `--limit` for a
quick look, and `--report` to write every number, broken down by question type and (for
typed-decisions) by workflow, to JSON. On a CUDA machine build with `-p:TorchSharpRuntime=TorchSharp-cuda-linux`.

Metric definitions follow laya's harness (`hard_metrics`), ported in `NLaya.Calibration.CalibrationMetrics`
and checked against it (`calibration.json` fixtures). The fine-tuning notebook's own typed-decisions
table uses a few different definitions: Brier against the teacher's soft labels, and a per-type confidence.
So its Brier (0.062) and ECE (0.213) aren't directly comparable with the harness block. Accuracy is the
same in both.

## Calibrate: fit temperatures on held-out data

Both base checkpoints ship over-confident, and `laya-multilingual` ships with all temperatures at 1.0.
Refitting is the single largest calibration fix available: laya reports mean ECE falling from 0.466
to 0.081 (`laya`) and from 0.314 to 0.106 (`laya-multilingual`). NLaya ports both of laya's fitting procedures:

| | Per type | Per bucket |
|---|---|---|
| From | the fine-tuning notebook (`fit_one_temp`) | the benchmark harness (`fit_temperature`) |
| Groups | choice, score, noul | type and option count: `choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2` |
| Search | exact minimum of cross-entropy over T in [0.1, 10] (`TemperatureFitMethod.Optimize`) | best of 160 geometric steps in [0.2, 10] (`TemperatureFitMethod.Grid`) |
| Needs | 10 samples per type | 25 per bucket (smaller buckets keep the per-type value) |
| Targets | the teacher distribution when the data has one, else the label | the label |
| Saved config | removes `temperature_by_options` | writes `temperature_by_options` |

Use per bucket when your questions vary in option count, and per type for one workflow with a fixed
question set. For the fine-tuning notebook's data, per type matches how the published checkpoint was
calibrated. The notebook's LBFGS (fixed step, no line search) usually lands within 0.2% of the minimum
but can stop short (see `CalibrationTests`). NLaya's exact search is never worse, so fitted
temperatures can differ slightly from a Python run on the same data.

From the harness:

```bash
# Fit on everything given and write a config for your own copy of the checkpoint
dotnet run -c Release --project benchmarks/NLaya.Eval -- --model convaiinnovations/laya-multilingual \
  --suite my_heldout.jsonl --refit per-bucket --save-config ./calibrated/rl_agent_config.json
```

From code:

```csharp
using NLaya.Calibration;

// Raw logits (before temperature) for held-out, labelled states
var logits = agent.PredictLogits(heldOut.Select(c => c.State), questions, new BatchOptions { BatchSize = 32 });
var samples = heldOut.Select((c, i) =>
    CalibrationSample.FromLabel(QuestionType.Choice, logits[i]["intent"], c.GoldIndex));

var fit = TemperatureFitter.FitPerBucket(samples, current: agent.Temperatures.PerType);
fit.SaveConfig(Path.Combine(agent.Directory, "rl_agent_config.json"), "./my-checkpoint/rl_agent_config.json");

// Or, for one language on the multilingual checkpoint, without touching the files:
o.LangTemperatures["de"] = fit.ToLanguageTemperature();

// And measure it
var report = CalibrationMetrics.Evaluate(heldOut.Select((c, i) =>
    new LabelledPrediction(c.GoldIndex, CalibrationMetrics.Softmax(logits[i]["intent"], fit.PerType[0]))));
```

Things to know:

- **Write a copy.** Checkpoints in the Hugging Face cache are links to shared blobs; `SaveConfig`
  refuses to write through a link. Copy the checkpoint directory, write its `rl_agent_config.json`, and
  load the copy with `Laya.LoadAsync("./my-checkpoint", ...)`.
- **Laya clamps temperatures to [0.5, 5] at load** (`TemperatureTable.Min`/`Max`), like the Python
  library. A fit below 0.5 (an under-confident model) is applied as 0.5; the harness prints a note
  when that happens.
- **Inherited buckets win over per-type temperatures.** `laya-typed-decisions/rl_agent_config.json`
  on the Hub still carries the base model's `temperature_by_options` (including `choice:11+ = 0.10`,
  clamped to 0.5). The notebook's current export removes them, and a per-type fit saved with
  `SaveConfig` does the same. Accuracy is unaffected, but ECE and Brier are.
- **Hold out properly.** Fit on data the model wasn't trained on, and report on a different slice
  (`--refit` does fit-on-half, report-on-half, as laya's harness does).
- **Gate on `AnswerConfidence`,** which is max(p), the quantity temperature scaling calibrates.
  `Confidence` (1 minus normalised entropy) isn't calibrated, and thresholds on it don't transfer
  between workflows.

## Fine-tune: to beat Jev on your own decisions

Fine-tuning is where the typed-decisions result (0.362 → 0.766) comes from. NLaya is inference-only:
train in Python with laya's notebook, then load the result in .NET. The output is an ordinary Laya
checkpoint directory, so nothing in NLaya changes.

1. **Build the dataset** in the typed-decisions shape (`LocalLLaMA/typed-decisions`): one row per case
   with `state` (text or JSON), `questions` (Python's dict shape, the same as `Questions.ToJson()`)
   and `gold`, where each question has a `label` and, ideally, teacher `probabilities`. Soft targets
   come from an LLM teacher or several annotators, and the training reward (a proper scoring rule)
   uses them. The published run used 1,200 cases (6,000 decisions); a few thousand decisions per
   workflow is a reasonable start.
2. **Train** with [`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
   (Kaggle's free 2×T4, about 4–5 hours for 4 epochs over ~30k questions). Point its dataset cell at your
   data and its `MODEL_ID` at the base checkpoint: `laya` for English, `laya-multilingual` for other
   languages or long inputs. Its settings, which reproduced 0.766:

   | Setting | Value |
   |---|---|
   | Method | RLCD: GRPO-style policy gradient on proper-scoring-rule rewards (spherical 0.75 + RPS 1.0), plus soft cross-entropy (weight 1.0) |
   | Epochs | 4, cosine LR to 1e-6 |
   | Learning rate | encoder 2.5e-5, head 1e-4, AdamW, weight decay 0.01, grad clip 1.0 |
   | Batch | 8 sequences per GPU × 2 GPUs × 4 accumulation steps = 64 |
   | Exploration | 4 samples per group; noise σ 0.4 → 0.1 over training |
   | Token budgets | `max_len` 1024, `head_max_len` 256 |
   | Precision | fp16 autocast, gradient checkpointing on encoder and head |
   | Calibration | 10% (at most 400 items) held out before training; one temperature per type |

3. **Load it in .NET**: `await Laya.LoadAsync("./laya_finetuned", o => o.UseTorchSharp())`. The notebook
   saves weights in float16, which NLaya reads and runs in float32 by default.
4. **Measure and calibrate** with `NLaya.Eval` on your held-out test split, as above. Compare against
   the base checkpoint and against whatever you're replacing (for example Jev's reported numbers, or
   your LLM teacher's agreement with itself, which the notebook reports as the ceiling).
5. **Keep it honest:** keep the test split out of training *and* calibration, and report per workflow
   and per question type (`--report` has both).

Why not train in .NET? TorchSharp could run the loop, but the training recipe is laya's, it changes
between releases, and there is no Python reference for NLaya to check a port against. The checkpoint
format is the contract: train where laya trains, run where you deploy.

## Design questions the model can answer

From laya's measured weaknesses:

- **Keep choices under ~20 options.** All options share `head_max_len` tokens (192 English, 256
  multilingual/typed-decisions), so 77 labels get about 4 tokens each and stop being distinguishable
  (Banking77: 0.425 vs Jev's 0.870). Split into a coarse and a fine question, raise `HeadMaxLen`, or
  shortlist first (laya's `predict_shortlist`, which isn't ported yet: embed, take the top 10–20,
  then ask).
- **Describe options.** `("billing", "invoices, payments, refunds")` beats a bare label.
- **Avoid boolean-word labels** (`yes`/`no`, `true`/`false`) for choice options; use meaningful or
  opaque labels.
- **`score` is the weakest primitive** (SST-5: 0.372), and the multilingual checkpoint rarely picks
  the first-listed level. Validate scores on your data, or use a choice when levels are really categories.
- **Long documents:** the multilingual checkpoint reads up to 8,192 tokens (`MaxLen = 8192`), but
  accuracy gets variable beyond about 4,000 tokens of preceding text. Put the decisive text first, or
  clean it (`EmailCleaner.State`).

## Make it fast

laya's reference points (Tesla T4, float32): 32.8 ms for one multilingual question and 39.5 ms for
English; 10 questions in one call take 72.3 ms (7.2 ms each); batched, about 1 ms per decision on
an RTX 5060 Ti. Python's optional fast path (bf16 weights, TileLang fused kernels, CUDA graphs)
claims 3.8× on single questions.

| Knob | Where | Effect |
|---|---|---|
| Device | `UseTorchSharp("cuda")` / `"mps"`, and a `TorchSharp-cuda-*` runtime package | the big one; `auto` picks CUDA when present |
| Precision | `UseTorchSharp(t => t.DType = ScalarType.BFloat16)` | about 2× less memory, faster on tensor cores; answers agree, probabilities within ~0.06 (checked by `NLAYA_DTYPE=bfloat16` parity) |
| Batching | `PredictBatch(states, questions, new BatchOptions { BatchSize = 32, SortByLength = true })` | amortises the pass; sorting cuts padding |
| Concurrent callers | `services.AddLaya(...).AddLayaMicroBatching()` or `new MicroBatchingPredictor(agent)` | turns many single requests into batches; tune `MaxDelay` (default 2 ms) and `MaxBatchSize` (32), and watch `laya.microbatch.size` |
| Long inputs | `MicroBatchOptions.PassSize` / `BatchOptions.BatchSize` | caps states per pass so long batches fit in GPU memory |
| First request | `agent.Warmup()` (the DI warm-up service calls it) | moves CUDA context and kernel selection out of the first request |
| Router | `Preload(...)`, `MaxLoaded` ≥ checkpoints in use | a switch between unloaded checkpoints reloads weights (7–10 s) |
| ONNX Runtime | `o.UseOnnx(dir, x => { x.UseCuda = true; })` with `Microsoft.ML.OnnxRuntime.Gpu` | on CUDA, hidden states stay on the device between `encoder.onnx` and `head.onnx` (I/O binding); set graph options with `x.Configure` |
| CPU | `TorchSharpOptions.NumThreads` (process-wide) | match physical cores; the multilingual checkpoint is ~3× faster than English on CPU |

What NLaya does for you on every pass:

- an unpadded batch (every single request) runs attention with no mask, so SDPA can pick its fused kernels
- sliding-window layers share one band mask instead of one copy per row
- RoPE tables are built once per theta
- the next chunk is tokenized while the current one runs

Measure latency with BenchmarkDotNet:

```bash
NLAYA_DEVICE=cuda dotnet run -c Release --project benchmarks/NLaya.Benchmarks -- --filter '*'
```

It times one question, ten questions, a 32-state batch and 32 concurrent micro-batched requests, per
checkpoint. Check GPU numerics with the parity suite on the device: `NLAYA_PARITY=1 NLAYA_DEVICE=cuda
NLAYA_DTYPE=bfloat16 dotnet test --project tests/NLaya.Parity -p:TorchSharpRuntime=TorchSharp-cuda-linux`
(or the manual `parity-gpu` workflow on a self-hosted GPU runner). The fixture fails if the backend
fell back to CPU.

## Measured with NLaya

Run on an Apple M4 Pro (24 GB) with TorchSharp on MPS in float32, using the suites from
`export_suites.py` (September 2026, laya commit `970dc8c`). "laya" is laya's own published number.

**Accuracy.** NLaya reproduces laya's numbers:

| Checkpoint | Suite | NLaya | laya | Jev |
|---|---|---|---|---|
| `laya-typed-decisions` | typed-decisions (2,000 decisions) | **0.766** | 0.766 | 0.727 |
| | &nbsp;&nbsp;invoice / security / customer service / agent traces | 0.804 / 0.766 / 0.764 / 0.730 | 0.804 / 0.766 / 0.764 / 0.730 | |
| | AG News (600) | 0.947 | 0.950 | 0.910 |
| | DAIR Emotion (600) | 0.583 | 0.595 | 0.480 |
| `laya` | typed-decisions | 0.361 | 0.362 | |
| | AG News / Emotion | 0.947 / 0.573 | | |
| `laya-multilingual` | XNLI German (300) | 0.790 | | |
| | AG News | 0.937 | | |

The typed-decisions results match laya exactly, per workflow too. AG News and Emotion are within about
one point; laya's table doesn't state which run or sample those two came from.

**Calibration** (ECE, lower is better; refits are fitted on half of each suite and measured on the other half):

| Checkpoint | Suite | Shipped | Refit |
|---|---|---|---|
| `laya-typed-decisions` | typed-decisions (per type, notebook style) | 0.207 | **0.147** |
| | AG News / Emotion (per bucket) | 0.152 / 0.269 | **0.037 / 0.140** |
| `laya` | typed-decisions / AG News / Emotion (per bucket) | 0.207 / 0.051 / 0.381 | **0.129 / 0.035 / 0.133** |
| `laya-multilingual` | XNLI German / AG News (per bucket) | 0.126 / 0.045 | **0.065 / 0.011** |

Over the whole typed-decisions suite, ECE at the shipped temperatures is 0.213, the figure laya's
model card reports. Refitting on all 2,000 answers gives per-type temperatures of about 1.15 (choice),
1.17 (score) and 1.13 (noul), against the shipped 1.01–1.06, whose inherited bucket overrides go
up to 1.98.

**Latency** (BenchmarkDotNet, `NLaya.Benchmarks`, MPS float32, mean per state):

| | `laya-multilingual` | `laya` (English) |
|---|---|---|
| One question | 11.1 ms | 25.6 ms |
| Ten questions, one call | 65.8 ms (6.6 ms each) | 144 ms (14.4 ms each) |
| 32 states, one `PredictBatch` | 6.2 ms per state | 14.1 ms per state |
| 32 concurrent callers via `MicroBatchingPredictor` | 6.2 ms per state | 14.1 ms per state |

Micro-batching gives concurrent single requests the same per-state cost as an explicit batch. On an M4
Pro, a single question is already faster than laya's T4 figures (32.8 ms / 39.5 ms). The eval harness's
own `--latency` p50 on real suite inputs was 15–47 ms, depending on the checkpoint and input length.

## Checklist

1. Route: English → `laya`, everything else → `laya-multilingual`, the typed-decisions workflows →
   `laya-typed-decisions`.
2. Keep choice questions under ~20 described options.
3. Measure on your own labelled data with `NLaya.Eval`.
4. Refit temperatures on held-out data (`--refit`, `--save-config`) and gate on `AnswerConfidence`.
5. If accuracy is short of what you need, fine-tune with laya's notebook on your decisions, then
   repeat steps 3–4 on the result.
6. For throughput: GPU, bf16, batching or `AddLayaMicroBatching`, warm-up and preloading.
