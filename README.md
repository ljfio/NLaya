# NLaya

.NET port of [`laya`](https://github.com/NandhaKishorM/laya), a non-autoregressive "System 1" decision
model. You give it a *state* (text, a JSON object or a conversation) and *typed questions* (`choice`,
`score` or `noul`). One forward pass returns typed answers with probabilities. It works with all three
Laya checkpoints:

| name | checkpoint | encoder | use it for |
|---|---|---|---|
| `english` | `convaiinnovations/laya` | ModernBERT-large | English |
| `multilingual` | `convaiinnovations/laya-multilingual` | mmBERT-base | 100+ languages, up to 8,192 tokens |
| `typed-decisions` | `convaiinnovations/laya-typed-decisions` | ModernBERT-large | the four typed-decisions workflows |

NLaya is a thin layer over Microsoft's ML stack:

| Concern | Library |
|---|---|
| Tokenization | `Microsoft.ML.Tokenizers` (`BpeTokenizer`, reading the checkpoint's `tokenizer.json`) |
| Tensor math | `System.Numerics.Tensors` |
| Inference (native) | `TorchSharp`: loads `model.safetensors` directly |
| Inference (ONNX) | `Microsoft.ML.OnnxRuntime`: runs laya's `encoder.onnx` + `head.onnx` export |

What NLaya adds on top is what the Python library defines: the ModernBERT forward pass and the decision
head, prompt layout, calibration, routing, language detection and email cleaning. The routing word
lists, email patterns and presets are embedded verbatim from the Python source
(see `tools/fixtures/LAYA_COMMIT`).

## Getting the models

NLaya doesn't download anything. Fetch checkpoints with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
into the standard cache, where NLaya (and the Python library) look them up by repo id:

```bash
hf download convaiinnovations/laya-multilingual        # one checkpoint
hf download convaiinnovations/laya                     # the bundle: english + multilingual/ + typed-decisions/ (what Router uses)
```

The cache location follows `HF_HUB_CACHE` / `HF_HOME` (default `~/.cache/huggingface/hub`), or set
`LayaOptions.CacheDir`. `Laya.LoadAsync` also accepts any local directory that holds a checkpoint.
If a checkpoint is missing, the error names the `hf download` command to run.

## Quickstart

```csharp
using NLaya;
using NLaya.TorchSharp;   // add TorchSharp-cpu (or TorchSharp-cuda-linux / -windows) to your app

await using var agent = await Laya.LoadAsync("convaiinnovations/laya-multilingual", o => o.UseTorchSharp());

var result = agent.Predict(
    new { body = "I was charged twice for invoice 4411. Please refund me today." },
    new Questions
    {
        ["department"] = Question.Choice("Which team should handle `body`?", new Dictionary<string, string?>
        {
            ["billing"] = "invoices, payments, refunds",
            ["technical"] = "bugs and outages",
            ["sales"] = "pricing",
        }),
        ["urgency"] = Question.Score("How urgent is this?", "not urgent", "somewhat urgent", "very urgent"),
        ["refund_requested"] = Question.Noul("Does the sender ask for money back?"),
    },
    new PredictOptions { MaxLen = 8192 });   // multilingual reads up to 8,192 tokens

Console.WriteLine(result.Choice("department").Choice);
Console.WriteLine(result.ToJsonString(indented: true)); // same shape as Python's result dict
```

Questions can also be given in the Python dict shape with `Questions.Parse(json)`. `PredictBatch`
answers the same questions for many states in shared forward passes.

## Router: pick the checkpoint per request

```csharp
using NLaya.Routing;

using var router = new Router(o => o.UseTorchSharp());
router.Preload(["english", "multilingual"]);                       // optional: no load at request time

var r = router.Predict("二重に請求されました", Presets.Triage());  // -> multilingual (non-Latin script)
Console.WriteLine($"{r.Routing!.Model}: {r.Routing.Reason}");

router.Predict(state, questions, new RouteOptions { Task = "typed_decisions" });   // explicit checkpoint
router.Predict(state, questions, new RouteOptions { LangGuess = s => myLid(s) });  // plug in your own language ID
```

Routing follows the Python rules: explicit `Model`, then `Task`, then `Lang`, then `LangGuess`, then
script and language detection (`NLaya.Lang.LanguageDetector`). `typed-decisions` is only chosen when you
ask for it, or when `AutoTaskDetection` is on and the question ids match one of its workflows.

## Presets, email and hooks

```csharp
var triage = Presets.Triage();   // also Email(), Guard(), Moderation(), Router()
var state = NLaya.Email.EmailCleaner.State(subject, rawBody, sender);   // strips quotes, signatures, disclaimers

agent.AddHook(LayaHooks.OnEnd(ctx => metrics.Record(ctx.Usage, ctx.ElapsedMs)));
```

A hook implements any of `ILayaHook`'s events: `OnPredictStart` (it may rewrite the input or call
`ctx.Skip(cached)`), `OnPredictEnd`, `OnError`, and for the Router `OnRoute`, `OnLoad` and `OnEvict`.

## ONNX Runtime

Export each checkpoint once with laya's script, then point NLaya at the output directory:

```bash
uv run --with laya --with onnx --with onnxruntime --with onnxscript \
  python laya-ts/scripts/export_onnx.py --repo convaiinnovations/laya-multilingual --out-dir ./onnx/multilingual
```

```csharp
using NLaya.Onnx;
await using var agent = await Laya.LoadAsync("./onnx/multilingual", o => o.UseOnnx("./onnx/multilingual"));
var router = new Router(new RouterOptions { ConfigureAgent = (name, o) => o.UseOnnx($"./onnx/{name}") });
```

## Tests

```bash
dotnet test --project tests/NLaya.Tests                    # tokenizers, prompts, routing, email vs Python (tokenizer cases need the models cached)
NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity    # all three checkpoints vs Python (needs: hf download convaiinnovations/laya and convaiinnovations/laya-multilingual)
NLAYA_PARITY=1 NLAYA_ONNX_ROOT=./onnx dotnet test --project tests/NLaya.Parity   # + ONNX (onnx/<name>/)
```

The golden fixtures come from the Python reference at the commit in `tools/fixtures/LAYA_COMMIT`.
`tools/fixtures/make_fixtures.py` regenerates them, along with the embedded tables.

License: Apache-2.0 (same as laya).
