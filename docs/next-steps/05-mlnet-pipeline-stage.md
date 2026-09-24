# 5. ML.NET pipeline stage (only if needed)

**Goal:** let ML.NET users score Laya questions over an `IDataView` as a pipeline step, for example
bulk-labelling a CSV of tickets, or feeding Laya answers into a downstream trainer as features.

**Do this only when there's a batch or data-science use case.** For LLM apps, step 1 (`NLaya.Extensions.AI`, done)
is the better integration. ML.NET's `IDataView` fixes its column schema when the pipeline is
built, so this only works for a **fixed question set per pipeline**, which is fine for the batch
use case.

## Shape

A `src/NLaya.ML` project referencing `Microsoft.ML` (core only; no TorchSharp or ONNX
transformers are needed, since the Laya backend already does inference).

```csharp
var pipeline = mlContext.Transforms.LayaPredict(agent, Presets.Triage(), inputColumnName: "Body");
var scored = pipeline.Fit(data).Transform(data);
// columns: intent (string), intent_probs (float[6]), is_urgent (float), frustration (float), ...
```

- **`LayaEstimator : IEstimator<LayaTransformer>`.** `Fit` is a no-op: nothing is trained.
  `GetOutputSchema` adds one column per question:
  - `choice`: the label (text) and a probabilities vector (`VBuffer<float>`, one slot per option,
    with slot names set to the labels).
  - `noul`: `P(true)` (single).
  - `score`: the expected score (single) and probabilities.
- **`LayaTransformer : ITransformer`.** Transform with a cursor that reads the input column in
  chunks and calls `agent.PredictBatch(chunk, questions, new BatchOptions { BatchSize = n })`, so
  forward passes are shared.
- **Saving.** `ITransformer` save/load can't embed the model. Either throw `NotSupportedException`
  from `Save`, or save only the model id and questions and reload through `HfCache`. Document which.
- **State columns.** Accept text columns. For multi-column states, take several input columns and
  build a JSON object state keyed by column name, matching how Python users pass dicts.

## Tests

- Build an `IDataView` from a list, run the transformer with a fake `ILayaBackendFactory` (fixed
  logits), and check the output schema, values and slot names.
- One parity test on a real checkpoint: the transformer's output equals `PredictBatch` for the same rows.
