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

## Microsoft.Extensions.AI

`NLaya.Extensions.AI` puts Laya in front of any `IChatClient` (OpenAI, Azure OpenAI, Ollama, ...). It's the
.NET counterpart of the Python LangChain integrations. Every decision is one `Predict` call on an
`ILayaPredictor`, which both `LayaAgent` and `Router` implement.

```csharp
using Microsoft.Extensions.AI;
using NLaya.Extensions.AI;

// Guardrail: screen each user turn (Presets.Guard() by default) before the LLM sees it.
IChatClient client = innerClient.AsBuilder()
    .UseLayaGuardrail(agent, o =>
    {
        o.Action = GuardrailAction.Filter;   // Raise (default) throws LayaGuardrailException; Annotate lets it through
        o.Thresholds["harm_severity"] = 2;   // see below
    })
    .Build();

// Router: pick a chat client per request with a Laya choice question.
var routed = new LayaRouterChatClient(agent, new LayaRouterChatClientOptions
{
    Routes =
    {
        ["simple"] = (smallModel, "greetings, FAQs, short factual questions"),
        ["complex"] = (largeModel, "multi-step reasoning, code, analysis"),
    },
    Fallback = "complex",
    ConfidenceThreshold = 0.6,
    ConfidenceMeasure = RouteConfidence.Answer,   // calibrated max(p); Entropy (the default) matches Python
});

// Tool: let an LLM call Laya inside its own agent loop.
var tools = new ChatOptions { Tools = [LayaTools.Triage(agent)] };
```

- **Guardrail violations** follow Python: a `noul` answer with P(true) ≥ threshold, or a `score` answer
  whose expected level is ≥ threshold (0.5 by default). Choice answers never count. With the default
  `Presets.Guard()`, `harm_severity` sits around 0.5 even for "When does the office open?", so set a
  per-question threshold such as `Thresholds["harm_severity"] = 2` ("serious").
- **Streaming** is checked once, before the first update.
- **Results travel on the response**, not the client: `ChatResponse.AdditionalProperties` holds
  `LayaChatProperties.Guardrail` (a `GuardrailResult`), `Route` and `Result`. On a stream they're on the
  first update.

`samples/NLaya.ChatGuardrail` runs all three against echo clients, so it needs no API key.

## Dependency injection

```csharp
builder.Services.AddLaya(Laya.MultilingualModel, o => o.UseTorchSharp());            // singleton LayaAgent + ILayaPredictor
builder.Services.AddKeyedLaya("english", Laya.DefaultModel, o => o.UseTorchSharp());  // several checkpoints
builder.Services.AddLayaRouter(o => o.ConfigureAgent = (_, a) => a.UseTorchSharp());  // singleton Router + ILayaPredictor
builder.Services.AddChatClient(innerClient).UseLayaGuardrail();                        // uses the registered ILayaPredictor
```

- Agents and the Router are singletons, and the container disposes them. `ILayaPredictor` resolves to
  the first of `AddLaya` / `AddLayaRouter` registered.
- Logging comes from the container's `ILoggerFactory`.
- A hosted service loads the model at startup, so the first request doesn't pay the 1–4 s load. For the
  Router it preloads `Laya:Preload`, or its default checkpoint. Turn it off with `"Laya": { "Warmup": false }`.
- `LayaSettings` binds from the `"Laya"` section (`"Laya:<key>"` for a keyed agent): `Model`, `Subfolder`,
  `Revision`, `CacheDir`, `Warmup`, and for the Router `MaxLoaded`, `Default`, `AutoTaskDetection`,
  `StandaloneRepos` and `Preload`. The backend stays in code, and values set in code win.

`samples/NLaya.WebApi` exposes `POST /triage` through the injected Router.

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

## Project layout

| Project | What it holds |
|---|---|
| `src/NLaya` | The API (`Laya`, `LayaAgent`, `Question`/`Questions`, `LayaResult`, `Router`, `Presets`) and the ported Python logic, one type per file: `Sequences/` builds the prompt, `Calibration/` handles temperature and decoding, `Lang/` detects script and language, `Email/` cleans email bodies, `Hooks/` runs lifecycle hooks, `Hub/` looks up the HF cache |
| `src/NLaya.TorchSharp` | `UseTorchSharp()`: the ModernBERT encoder and Laya decision head as TorchSharp ops, weights read from `model.safetensors` |
| `src/NLaya.Onnx` | `UseOnnx(dir)`: runs `encoder.onnx` then `head.onnx` with ONNX Runtime |
| `src/NLaya.Extensions.AI` | `IChatClient` guardrail and router, `AIFunction` tools, and `AddLaya` / `AddLayaRouter` registration |
| `tests/NLaya.Tests` | Parity with Python for tokenization, prompts, JSON, routing and email (golden fixtures), and the Extensions.AI middleware and DI with fakes |
| `tests/NLaya.Parity` | Model parity for all three checkpoints on both backends, and an end-to-end guardrail test |
| `tools/fixtures` | Regenerates the fixtures and the embedded tables from the Python reference |
| `samples/NLaya.Quickstart` | A single agent, then the Router across all three checkpoints |
| `samples/NLaya.ChatGuardrail` | Guardrail, router and tool over echo chat clients (no API key) |
| `samples/NLaya.WebApi` | Minimal API: `POST /triage` through the DI-registered Router |

A backend only turns a padded batch of token ids into logits (`ILayaBackend.Run`). Everything
before and after that, including tokenization, prompt layout, calibration and decoding, lives in
`NLaya`, so both backends give identical answers.

## Tests

```bash
dotnet test --project tests/NLaya.Tests                    # tokenizers, prompts, routing, email vs Python (tokenizer cases need the models cached)
NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity    # all three checkpoints vs Python (needs: hf download convaiinnovations/laya and convaiinnovations/laya-multilingual)
NLAYA_PARITY=1 NLAYA_ONNX_ROOT=./onnx dotnet test --project tests/NLaya.Parity   # + ONNX (onnx/<name>/)
```

The golden fixtures come from the Python reference at the commit in `tools/fixtures/LAYA_COMMIT`.
`tools/fixtures/make_fixtures.py` regenerates them, along with the embedded tables.

## Packages and releases

| Package | Adds |
|---|---|
| `NLaya` | The core library; no native code |
| `NLaya.TorchSharp` | The TorchSharp backend. The app adds `TorchSharp-cpu`, `TorchSharp-cuda-linux` or `TorchSharp-cuda-windows` |
| `NLaya.Onnx` | The ONNX Runtime backend (CPU). Add `Microsoft.ML.OnnxRuntime.Gpu` for CUDA |
| `NLaya.Extensions.AI` | `Microsoft.Extensions.AI` middleware and dependency injection |

Versions come from git tags through [MinVer](https://github.com/adamralph/minver): tag `v0.1.0` builds
`0.1.0`, and untagged commits build `0.1.0-alpha.0.<height>`. GitHub Actions runs:

- **`build.yml`**: on every push and PR, on Ubuntu and macOS, builds and runs `tests/NLaya.Tests` (with the tokenizers downloaded).
- **`parity.yml`**: nightly and on demand, downloads the checkpoints and runs model parity (TorchSharp).
- **`release.yml`**: on a `v*` tag, tests, packs, pushes to nuget.org with
  [trusted publishing](https://learn.microsoft.com/nuget/nuget-org/trusted-publishing) (no stored API key;
  it needs a `NUGET_USER` secret with the nuget.org profile name), and creates a GitHub release.

## Native AOT

All four packages are marked `IsAotCompatible`, so the trim and AOT analyzers fail the build on
reflection-based code. Apps can publish with `<PublishAot>true</PublishAot>`: both backends run under
Native AOT and give the same answers as a JIT build (ONNX Runtime and TorchSharp load their native
libraries from the app folder; TorchSharp itself reports IL3000 warnings about `Assembly.Location`
but still loads).

Pass states through source-generated JSON metadata rather than reflection:

```csharp
[JsonSerializable(typeof(Ticket))]
internal partial class AppJsonContext : JsonSerializerContext;

var state = LayaState.From(ticket, AppJsonContext.Default.Ticket);        // AOT-safe
var chat = LayaState.Conversation(turns, AppJsonContext.Default.Turn);    // AOT-safe
agent.Predict(state, questions);
```

`LayaState.From(object)`, `LayaState.Conversation(IEnumerable<object?>)` and `Predict(object, ...)`
still work in JIT apps but are marked `[RequiresUnreferencedCode]` / `[RequiresDynamicCode]`, so a
Native AOT publish flags them. `AddLaya` binds `LayaSettings` with the configuration binding source
generator, so DI works under AOT too.

`samples/NLaya.AotSmoke` is published with Native AOT in CI (`build.yml`, job `aot`): it checks
questions, JSON states, tokenization, decoding and DI binding with a fake backend, and with
`--onnx <dir>` runs a real prediction.

## Roadmap

Planned work, with the context needed to pick each item up, is in [`docs/next-steps/`](docs/next-steps/README.md):
`Decide<T>()` typed schemas, publishing the ONNX exports, closing tokenizer gaps upstream, and an
optional ML.NET pipeline stage.

License: Apache-2.0 (same as laya). NLaya is a port of [laya](https://github.com/NandhaKishorM/laya) by
Convai Innovations / NandhaKishorM, and the model weights are theirs.
