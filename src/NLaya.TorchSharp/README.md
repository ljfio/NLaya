# NLaya.TorchSharp

TorchSharp inference backend for [NLaya](https://www.nuget.org/packages/NLaya): runs the ModernBERT
encoder and Laya decision head, reading `model.safetensors` directly.

This package references the managed `TorchSharp` package only. **Add a libtorch runtime to your app**:
`TorchSharp-cpu`, `TorchSharp-cuda-linux` or `TorchSharp-cuda-windows`.

```csharp
await using var agent = await Laya.LoadAsync(Laya.DefaultModel, o => o.UseTorchSharp("cpu")); // or "cuda", "mps", "auto"
```

NLaya is a .NET port of [laya](https://github.com/NandhaKishorM/laya) by Convai Innovations / NandhaKishorM (Apache-2.0). The Laya model weights are theirs, published on Hugging Face under [convaiinnovations](https://huggingface.co/convaiinnovations). Source, docs and samples: https://github.com/ljfio/NLaya.
