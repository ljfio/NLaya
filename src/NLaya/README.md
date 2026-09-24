# NLaya

Typed, calibrated "System 1" decisions with the Laya models. Give it a state (text, JSON or a
conversation) and typed questions (`choice`, `score`, `noul`); one forward pass returns typed answers
with probabilities. Includes the `Router` (picks the English, multilingual or typed-decisions
checkpoint per request), language detection, email cleaning and question presets.

This package has no inference code of its own. Add a backend:

- **NLaya.TorchSharp**: loads `model.safetensors` directly.
- **NLaya.Onnx**: runs laya's ONNX export with ONNX Runtime.

NLaya doesn't download models. Fetch them with the Hugging Face CLI first:

```bash
hf download convaiinnovations/laya-multilingual
```

```csharp
using NLaya;
using NLaya.TorchSharp;

await using var agent = await Laya.LoadAsync(Laya.MultilingualModel, o => o.UseTorchSharp());
var result = agent.Predict("I was charged twice, please refund me", Presets.Triage());
Console.WriteLine(result.Choice("intent").Choice);
```

NLaya is a .NET port of [laya](https://github.com/NandhaKishorM/laya) by Convai Innovations / NandhaKishorM (Apache-2.0). The Laya model weights are theirs, published on Hugging Face under [convaiinnovations](https://huggingface.co/convaiinnovations). Source, docs and samples: https://github.com/ljfio/NLaya.
