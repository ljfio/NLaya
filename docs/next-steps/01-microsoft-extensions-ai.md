# 1. Microsoft.Extensions.AI middleware

**Goal:** make Laya a drop-in part of any .NET LLM app built on `Microsoft.Extensions.AI`, which is
Microsoft's provider-neutral `IChatClient` abstraction over OpenAI, Azure OpenAI, Ollama and others.
This is the .NET version of the Python library's LangChain integrations
(`laya/integrations/langchain.py`: `LayaRouter`, `LayaGuardrail`, `LayaTriage`, `LayaEvaluator`).
It is the most direct answer to "lightweight integration with Microsoft's AI libraries".

## Shape

A new project, `src/NLaya.Extensions.AI`, that references `NLaya` and `Microsoft.Extensions.AI`
(the `ChatClientBuilder` extensions live there, and the types in `Microsoft.Extensions.AI.Abstractions`).
It stays thin: every decision is a `Predict` call on an existing `LayaAgent` or `Router`.

```csharp
IChatClient client = new OpenAIClient(key).GetChatClient("gpt-4o-mini").AsIChatClient()
    .AsBuilder()
    .UseLayaGuardrail(agent)                                   // checks the user turn with Presets.Guard() first
    .Build();

var routed = new LayaRouterChatClient(agent, new()
{
    ["simple"]  = (smallModel, "greetings, FAQs, short factual questions"),
    ["complex"] = (largeModel, "multi-step reasoning, code, analysis"),
}, fallback: "complex", confidenceThreshold: 0.6);
```

### `LayaGuardrailChatClient : DelegatingChatClient`

Port of `LayaGuardrail.invoke`:

- **Input:** the text of the latest user message (`ChatMessage.Text`). Allow a `Func` to pick other
  text, as Python's `state_key` does.
- **Questions:** `Presets.Guard()` by default, or the caller's own.
- **Violations:** a `noul` answer with `Noul >= threshold`, or a `score` answer with
  `Score >= threshold`. The default threshold is 0.5. Python applies the same threshold to both,
  which for `score` means an expected level of 0.5 or more; keep that behaviour, but document it
  and allow per-question thresholds.
- **Actions** (Python's `action`):
  - `raise` (default): throw `LayaGuardrailException` with `Violations` and the raw `LayaResult`.
  - `filter`: return a `ChatResponse` holding a rejection message; the inner client is not called.
  - `annotate`: call the inner client and attach the guardrail result to `ChatResponse.AdditionalProperties`.
- **Streaming:** check once before streaming starts, so `GetStreamingResponseAsync` is guarded the
  same way.

### `LayaRouterChatClient : IChatClient`

Port of `LayaRouter.invoke`:

- It asks one `choice` question (id `route`) built from label → description. The default
  instructions are `"Which route should handle this request?"`.
- It picks the inner `IChatClient` for the chosen label.
- **Confidence gating:** Python compares the answer's `confidence` (the normalized-entropy value,
  which is *not* calibrated) against `confidence_threshold`, and falls back when it is lower.
  Decide whether to mirror that or gate on `AnswerConfidence` (calibrated `max(p)`), and document
  the choice. Offering both is cheap.
- Expose the last decision, like Python's `last_decision`, through `ChatResponse.AdditionalProperties`
  (for example `"laya.route"` and `"laya.result"`), not as mutable state, because clients are shared
  across threads.

### Optional

- **`AIFunction` tools.** `AIFunctionFactory.Create((string text) => agent.Predict(text, questions))`
  lets an LLM call Laya as a tool, for triage or classification inside an agent loop. A
  `LayaTools.Triage(agent)` helper would be a few lines.
- **`LayaEvaluator`.** Grading LLM output against a rubric (`noul`/`score` questions over the
  response). It fits as a `DelegatingChatClient` that annotates responses. It's lower priority.

## Enabling change in `NLaya`: an `ILayaPredictor` interface

The middleware should accept either a `LayaAgent` or a `Router`, and tests need a fake. Add a small
interface in `NLaya`, implement it on both types, and have the middleware take it:

```csharp
public interface ILayaPredictor
{
    LayaResult Predict(LayaState state, Questions questions, PredictOptions? options = null);
    Task<LayaResult> PredictAsync(LayaState state, Questions questions, PredictOptions? options = null, CancellationToken ct = default);
}
```

`Router.Predict` takes `RouteOptions`, which derives from `PredictOptions`, so accept
`PredictOptions` and pattern-match inside `Router`.

## Tests

- **Unit tests with no model:** a fake `ILayaPredictor` returning canned `LayaResult`s, and a fake
  inner `IChatClient`. Check each guardrail action, router selection, fallback and threshold, and
  that the inner client is not called when a request is filtered.
- **One opt-in end-to-end test** in `tests/NLaya.Parity` (`NLAYA_PARITY=1`): a real agent with an
  obviously unsafe prompt, and an echo `IChatClient`.

## Done when

- `NLaya.Extensions.AI` builds, and its README section shows guardrail, router and tool usage.
- The Quickstart sample (or a new `samples/NLaya.ChatGuardrail`) guards a fake echo chat client, so
  it runs without an API key.
