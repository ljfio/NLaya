# NLaya.ML

[NLaya](https://www.nuget.org/packages/NLaya) as an ML.NET pipeline step: answer Laya questions about
every row of an `IDataView`, for example to bulk-label a CSV of tickets or to feed calibrated answers
into a trainer as features.

```csharp
var ml = new MLContext();
var data = ml.Data.LoadFromTextFile<Ticket>("tickets.csv", separatorChar: ',', hasHeader: true);

var scored = ml.Transforms.Laya(agent, Presets.Triage(), "Body").Fit(data).Transform(data);
// intent (text), intent_probs (float[6], slot names = labels), intent_confidence, is_urgent (bool), ...
```

For each question id it adds the answer (`choice`: the label; `score`: the expected score; `noul`:
true/false), `id_probs` (choice and score, one named slot per option) or `id_probability` (noul,
P(true)), and `id_confidence`. One input column is the state's text; several become a JSON object
keyed by column name. Rows are answered lazily, `ChunkSize` rows (256) per `PredictBatch` call.

The predictor can be a `LayaAgent` or a `Router`. The transformer can't be saved (the model isn't in
the ML.NET file) and has no `PredictionEngine`; call the predictor directly for single rows.

NLaya is a .NET port of [laya](https://github.com/NandhaKishorM/laya) by Convai Innovations / NandhaKishorM (Apache-2.0). The Laya model weights are theirs, published on Hugging Face under [convaiinnovations](https://huggingface.co/convaiinnovations). Source, docs and samples: https://github.com/ljfio/NLaya.
