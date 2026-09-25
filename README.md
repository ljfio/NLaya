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

var questions = new Questions()
    .Choice("department", "Which team should handle `body`?",
        ("billing", "invoices, payments, refunds"),
        ("technical", "bugs and outages"),
        ("sales", "pricing"))
    .Score("urgency", "How urgent is this?", "not urgent", "somewhat urgent", "very urgent")
    .Noul("refund_requested", "Does the sender ask for money back?");

var result = agent.Predict(
    new { body = "I was charged twice for invoice 4411. Please refund me today." },
    questions,
    new PredictOptions { MaxLen = 8192 });   // multilingual reads up to 8,192 tokens

Console.WriteLine(result.Choice("department").Choice);
Console.WriteLine(result.ToJsonString(indented: true)); // same shape as Python's result dict
```

`PredictBatch` answers the same questions for many states in shared forward passes.

### Defining questions

The fluent form above, a collection initializer and Python's dict shape all build the same
`Questions`, so use whichever reads best (or mix them):

```csharp
// Options built in a loop, and a yes/no question with descriptions and display labels
var q = new Questions()
    .Choice("product", "Which product is `body` about?", c =>
    {
        foreach (var p in catalog) c.Option(p.Sku, p.Name);
    })
    .Score("frustration", "How frustrated is the sender?", s => s.Level("calm").Level("annoyed").Level("angry"))
    .Noul("needs_reply", "Does the sender expect a reply?", n => n
        .WhenTrue("they asked a question or requested an action")
        .Labels("no", "yes"));

// Collection initializer, as in Python's dict literal
var q2 = new Questions
{
    ["refund_requested"] = Question.Noul("Does the sender ask for money back?"),
    ["department"] = Question.Choice("Which team?", ("billing", "invoices"), ("other", null)),
};

// Python's dict shape, verbatim
var q3 = Questions.Parse("""{"refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"}}""");
```

A repeated id in a fluent chain throws; the indexer replaces the earlier question, like a Python
dict. Read answers with `result.Choice("id")`, `result.Score("id")`, `result.Noul("id")`, or
`result.TryGet<NoulAnswer>("id", out var a)` when a hook may have dropped one. A choice has
`Choice` and `Probabilities` by label, a score has `Score` (the expected level), `Probabilities` by
level and `MostLikelyLevel`, and a noul has `Value` and `Probability` (P(true)). `ToJson()` writes
Python's result dict, including its fixed `"model": "laya-rl-agent"`; `result.Model` is the model id.

## Typed decisions

Describe the answer as a C# type and get one back (port of Python's `decide` / `laya.structured`).
Each property becomes one question: an enum a choice, a `bool` a noul, and an integer with `[Range]`
(at most 10 levels) a score. The property name is the question id; `[Description]` is the instructions.

```csharp
public enum Team
{
    [Description("Payments, invoices and refunds")] Billing,   // enum member descriptions: a .NET extra
    [Description("Bugs and outages")] Support,
    Other,
}

public sealed record Triage(
    [property: Description("Which team should handle `body`?")] Team Team,
    [property: Description("How urgent is this?"), Range(1, 3)] int Urgency,
    bool NeedsHuman);

[JsonSourceGenerationOptions(UseStringEnumConverter = true)]   // enums must serialize as strings
[JsonSerializable(typeof(Triage))]
internal partial class AppJson : JsonSerializerContext;

Triage t = agent.Decide(email, AppJson.Default.Triage);                     // AOT-safe
DecisionResult<Triage> d = agent.DecideWithDetails(email, AppJson.Default.Triage);
// d.Value, d.Confidence["Team"], d.Probabilities["Urgency"], d.Answers, d.Usage, d.Routing

Triage quick = agent.Decide<Triage>(email);   // reflection: fine in normal apps, warns under trimming / native AOT
JsonObject values = agent.Decide(email, JsonNode.Parse(schemaJson)!);   // Python-style: a raw JSON schema
```

`Decide` works on `LayaAgent`, `Router` and any `ILayaPredictor`, with `...Async` twins. Plans are
cached per type; `DecisionSchema.For(...)` / `DecisionSchema.FromJson(...)` expose the questions and
`Project(...)` (Python's `answers_to_json`). A score decides its most likely level, not the rounded
expected score. Free strings, arrays, nested objects and `$ref` throw `LayaSchemaException` with
Python's message.

Enum member descriptions are the one difference from Python, whose schemas can't describe enum
values: they become the options' descriptions. In a raw JSON schema, write them the standard way,
`"oneOf": [{"const": "billing", "description": "..."}, ...]`. Enums without descriptions, and every
schema Python accepts, give exactly Python's questions.

## Router: pick the checkpoint per request

```csharp
using NLaya.Routing;

using var router = new Router(o => o.UseTorchSharp());
router.Preload([Checkpoint.English, Checkpoint.Multilingual]);    // optional: no load at request time

var r = router.Predict("二重に請求されました", Presets.Triage());  // -> multilingual (non-Latin script)
Console.WriteLine($"{r.Routing!.Checkpoint}: {r.Routing.Reason}");

router.Predict(state, questions, new RouteOptions { Checkpoint = Checkpoint.TypedDecisions });  // explicit
router.Predict(state, questions, new RouteOptions { LangGuess = s => myLid(s) });  // plug in your own language ID
```

Routing follows the Python rules: an explicit `Checkpoint` (Python's `model=` and `task=`), then `Lang`,
then `LangGuess`, then script and language detection (`NLaya.Lang.LanguageDetector`). `Checkpoints.Parse`
reads Python's names and aliases ("english", "ml", "typed_decisions"), and `checkpoint.Name()` gives them back. `typed-decisions` is only chosen when you
ask for it, or when `AutoTaskDetection` is on and the question ids match one of its workflows.
Checkpoints load on first use, outside the router's lock: requests for a checkpoint that is already
loaded keep flowing while another loads, and concurrent requests for the same one share its load.

## Batch and streaming

```csharp
// Many states, one question set: shared forward passes, results in input order.
var results = agent.PredictBatch(states, Presets.Triage(), new BatchOptions { BatchSize = 16, SortByLength = true });

// Any length of input (a DB cursor, a CSV, a queue): read and scored a chunk at a time, memory stays bounded.
await foreach (var (ticket, result) in agent.PredictStreamAsync(tickets, t => t.Body, Presets.Triage(), ct: ct))
    await SaveAsync(ticket.Id, result.Choice("intent").Choice);

// Mixed requests through the Router: each checkpoint loads once, same-question requests share passes.
var routed = router.PredictBatch(
[
    new RouteRequest("I was charged twice", Presets.Triage()),
    new RouteRequest("二重に請求されました", Presets.Triage()),
    new RouteRequest(incident, securityQuestions) { Checkpoint = Checkpoint.TypedDecisions },
]);
```

`PredictStreamAsync` works on anything that implements `ILayaPredictor` (an agent, a router, a test
fake). It reads `BatchSize` items at a time (default 32, `BatchOptions.DefaultStreamBatchSize`) and
answers each chunk with one `PredictBatch` call, so hooks see one call per chunk. It checks cancellation
between chunks. `Router.RouteBatch` / `PredictBatch` port Python's `route_batch` / `predict_batch`:
requests are grouped by checkpoint, then by question set, and Router hooks still run per request.
`Router.PredictStreamAsync` streams `RouteRequest`s the same way.

## ML.NET

`NLaya.ML` adds Laya as a pipeline step over an `IDataView` (a CSV, a database query, any ML.NET data):

```csharp
var ml = new MLContext();
var scored = ml.Transforms.Laya(agent, Presets.Triage(), "Body").Fit(data).Transform(data);

foreach (var row in ml.Data.CreateEnumerable<TriagedTicket>(scored, reuseRowObject: false))
    Console.WriteLine($"{row.Id}: {row.Intent} ({row.IntentConfidence:P0}), urgent: {row.IsUrgent}");

public sealed class TriagedTicket
{
    public int Id { get; set; }
    [ColumnName("intent")] public string Intent { get; set; } = "";
    [ColumnName("intent_confidence")] public float IntentConfidence { get; set; }
    [ColumnName("is_urgent")] public bool IsUrgent { get; set; }
}
```

Each question adds typed columns: the answer under its id (`choice` → text label, `score` → expected
score, `noul` → boolean), `id_probs` (a vector with one slot per option, named by label) or
`id_probability` (P(true)), and `id_confidence`. Pass several input columns to build a JSON object
state keyed by column name. `LayaTransformerOptions` sets the chunk size (rows per `PredictBatch`
call, default 256), the forward-pass batch and a column-name prefix.

Rows are answered lazily, and only when an answer column is read. Every cursor over the answer
columns runs the model, so to read the output more than once, cache it with every column prefetched:
`ml.Data.Cache(scored, [.. scored.Schema.Select(c => c.Name)])`. The transformer isn't saved with the
ML.NET model (the Laya model lives in the predictor) and has no `PredictionEngine`. `Microsoft.ML`
isn't AOT-compatible, so neither is this package.

## Presets, email and hooks

```csharp
var triage = Presets.Triage();   // also Email(), Guard(), Moderation(), Router()
var state = NLaya.Email.EmailCleaner.State(subject, rawBody, sender);   // strips quotes, signatures, disclaimers

agent.AddHook(LayaHooks.OnEnd(ctx => metrics.Record(ctx.Usage, ctx.Elapsed)));
builder.Services.AddLayaHook<AuditHook>();   // with DI: every agent and router the container builds
```

A hook implements any of `ILayaHook`'s events: `OnPredictStart` (it may rewrite the input or call
`ctx.Skip(cached)`), `OnPredictEnd`, `OnError`, and for the Router `OnRoute`, `OnLoad` and `OnEvict`.
A failing hook throws unless `ThrowOnHookError` is false, when it is logged and skipped. Prefer
`AddLayaHook` over the process-wide `LayaHooks.SetDefaults` in hosted apps.

## Tracing and metrics

NLaya reports through the standard .NET diagnostics APIs (`ActivitySource` and `Meter`, both named
`"NLaya"`), so OpenTelemetry or `dotnet-counters` pick it up with no extra package:

```csharp
builder.Services.AddOpenTelemetry()
    .WithTracing(t => t.AddSource(LayaTelemetry.Name))
    .WithMetrics(m => m.AddMeter(LayaTelemetry.Name));
```

Spans are `laya.load` and `laya.predict`. Metrics are `laya.load.duration`, `laya.predict.duration`
(seconds), `laya.predict.states`, `laya.predict.tokens` and `laya.route.decisions`, all tagged with
`laya.model`. Failures set the span status and an `error.type` tag. Hooks remain the place for
per-call logic such as caching or redaction.

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
- Logging comes from the container's `ILoggerFactory`, and hooks from `AddLayaHook<T>()` / `AddLayaHook(hook)`.
- A hosted service loads the model at startup, so the first request doesn't pay the 1–4 s load. For the
  Router it preloads `Laya:Preload`, or its default checkpoint. Turn it off with `"Laya": { "Warmup": false }`.
- `LayaSettings` binds from the `"Laya"` section (`"Laya:<key>"` for a keyed agent): `Model`, `Subfolder`,
  `Revision`, `CacheDir`, `Warmup`, and for the Router `MaxLoaded`, `Default`, `AutoTaskDetection`,
  `StandaloneRepos` and `Preload` (checkpoint names or aliases). The backend stays in code, and values set in code win.

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
var router = new Router(new RouterOptions { ConfigureAgent = (checkpoint, o) => o.UseOnnx($"./onnx/{checkpoint.Name()}") });
```

ONNX Runtime's own usage telemetry is switched off unless you set `OnnxOptions.EnableTelemetry`. Its
upload at process exit could also crash the process on macOS (see `docs/next-steps`).

## Project layout

| Project | What it holds |
|---|---|
| `src/NLaya` | The API (`Laya`, `LayaAgent`, `Question`/`Questions`, `LayaResult`, `Router`, `Presets`) and the ported Python logic, one type per file: `Sequences/` builds the prompt, `Calibration/` handles temperature and decoding, `Lang/` detects script and language, `Email/` cleans email bodies, `Hooks/` runs lifecycle hooks, `Hub/` looks up the HF cache |
| `src/NLaya.TorchSharp` | `UseTorchSharp()`: the ModernBERT encoder and Laya decision head as TorchSharp ops, weights read from `model.safetensors` |
| `src/NLaya.Onnx` | `UseOnnx(dir)`: runs `encoder.onnx` then `head.onnx` with ONNX Runtime |
| `src/NLaya.Extensions.AI` | `IChatClient` guardrail and router, `AIFunction` tools, and `AddLaya` / `AddLayaRouter` registration |
| `src/NLaya.ML` | `mlContext.Transforms.Laya(...)`: an ML.NET estimator/transformer that adds answer columns to an `IDataView` |
| `tests/NLaya.Tests` | Parity with Python for tokenization, prompts, JSON, routing and email (golden fixtures), and the Extensions.AI middleware and DI with fakes |
| `tests/NLaya.Parity` | Model parity for all three checkpoints on both backends, and an end-to-end guardrail test |
| `tests/NLaya.Testing` | Shared by both test projects: the golden fixtures from `make_fixtures.py`, fixture loading and JSON comparison helpers |
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
| `NLaya.ML` | An ML.NET pipeline stage (`mlContext.Transforms.Laya`) |

Versions come from git tags through [MinVer](https://github.com/adamralph/minver): tag `v0.1.0` builds
`0.1.0`, and untagged commits build `0.1.0-alpha.0.<height>`. GitHub Actions runs:

- **`build.yml`**: on every push and PR, on Ubuntu and macOS, builds and runs `tests/NLaya.Tests` (with the tokenizers downloaded).
- **`parity.yml`**: nightly and on demand, downloads the checkpoints and runs model parity (TorchSharp).
- **`release.yml`**: on a `v*` tag, tests, packs, pushes to nuget.org with
  [trusted publishing](https://learn.microsoft.com/nuget/nuget-org/trusted-publishing) (no stored API key;
  it needs a `NUGET_USER` secret with the nuget.org profile name), and creates a GitHub release.

## Native AOT

`NLaya`, `NLaya.TorchSharp`, `NLaya.Onnx` and `NLaya.Extensions.AI` are marked `IsAotCompatible`, so the trim and AOT analyzers fail the build on
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
publishing the ONNX exports and closing tokenizer gaps upstream.

License: Apache-2.0 (same as laya). NLaya is a port of [laya](https://github.com/NandhaKishorM/laya) by
Convai Innovations / NandhaKishorM, and the model weights are theirs.
