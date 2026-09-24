# NLaya.Onnx

ONNX Runtime inference backend for [NLaya](https://www.nuget.org/packages/NLaya): runs laya's
`encoder.onnx` + `head.onnx` export (made with laya's `export_onnx.py`).

It uses the CPU `Microsoft.ML.OnnxRuntime` package. For CUDA, add `Microsoft.ML.OnnxRuntime.Gpu` to your app.

```csharp
await using var agent = await Laya.LoadAsync("./onnx/multilingual", o => o.UseOnnx("./onnx/multilingual"));
```

NLaya is a .NET port of [laya](https://github.com/NandhaKishorM/laya) by Convai Innovations / NandhaKishorM (Apache-2.0). The Laya model weights are theirs, published on Hugging Face under [convaiinnovations](https://huggingface.co/convaiinnovations). Source, docs and samples: https://github.com/ljfio/NLaya.
