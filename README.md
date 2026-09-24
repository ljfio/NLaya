# NLaya

.NET port of [`laya`](https://github.com/NandhaKishorM/laya): a non-autoregressive "System 1" decision
model. Give it a state (text, JSON or a conversation) and typed questions (`choice`, `score`, `noul`);
it returns typed answers with probabilities from a single forward pass.

NLaya is a thin layer over Microsoft's ML stack:

| Concern | Library |
|---|---|
| Tokenization | `Microsoft.ML.Tokenizers` (`BpeTokenizer`, reading the checkpoint's `tokenizer.json`) |
| Tensor math | `System.Numerics.Tensors` |
| Inference (native) | `TorchSharp`: loads `model.safetensors` directly |
| Inference (ONNX) | `Microsoft.ML.OnnxRuntime`: runs laya's `encoder.onnx` + `head.onnx` export |

## Quickstart

```csharp
using NLaya;
using NLaya.TorchSharp;   // add TorchSharp-cpu (or a CUDA runtime) to your app

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
        ["refund_requested"] = Question.Noul("Does the sender ask for money back?"),
    });

Console.WriteLine(result.Choice("department").Choice);
Console.WriteLine(result.ToJsonString(indented: true)); // same shape as Python's result dict
```

Checkpoints download into the standard Hugging Face cache, so they are shared with the Python library.

### ONNX Runtime

Export once with laya's script, then point NLaya at the output directory:

```bash
uv run --with laya --with onnx --with onnxruntime --with onnxscript \
  python laya-ts/scripts/export_onnx.py --repo convaiinnovations/laya-multilingual --out-dir ./model-ml
```

```csharp
using NLaya.Onnx;
await using var agent = await Laya.LoadAsync("./model-ml", o => o.UseOnnx("./model-ml"));
```

## Tests

```bash
dotnet test --project tests/NLaya.Tests                       # unit + tokenizer/sequence parity (downloads tokenizers)
NLAYA_PARITY=1 dotnet test --project tests/NLaya.Parity       # model parity vs Python (downloads the checkpoint)
NLAYA_PARITY=1 NLAYA_ONNX_DIR=./model-ml dotnet test --project tests/NLaya.Parity
```

Golden fixtures come from the Python reference: `uv run --python 3.12 --with laya --with tokenizers tools/fixtures/make_fixtures.py`.

License: Apache-2.0 (same as laya).
