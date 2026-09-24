# 4. Less custom code

Independent tasks that shrink what NLaya has to maintain. Each can be done on its own.

## 4a. Target .NET 10 only (done)

All projects target `net10.0`. `OrderedMap<T>` became `System.Collections.Generic.OrderedDictionary<string, T>`
(`Questions` derives from it, and results expose it), the Router and the TorchSharp rope cache use
`System.Threading.Lock`, and `Microsoft.Extensions.Logging.Abstractions` / `System.Numerics.Tensors`
moved to 10.0.x. No `net8`-only workarounds were found.

## 4b. Publish the ONNX exports to the Hub

Today ONNX users must run laya's Python `export_onnx.py` (commands in [README](README.md)). Instead:

- Upload `onnx/<name>/` to a Hugging Face repo per checkpoint under the user's account. Each holds
  `encoder.onnx`, `encoder.onnx.data`, `head.onnx`, `head.onnx.data`, `tokenizer.json` and
  `rl_agent_config.json`. Name them, for example, `<user>/laya-multilingual-onnx`. The model
  license is Apache-2.0, so redistribution is allowed with attribution; say so in each model card,
  link the source checkpoint, and record the laya commit and export date. Ask the user before
  publishing anything: the Hub is public by default.
- Let `UseOnnx` take a Hub id (resolved by `HfCache.Snapshot`) as well as a directory:
  `o.UseOnnx("<user>/laya-multilingual-onnx")` after `hf download <user>/laya-multilingual-onnx`.
- **Optional:** export `float16` / quantized (int8) variants with `onnxruntime`'s quantization tools
  for smaller CPU deployments, and check parity with a looser tolerance.

## 4c. Close the `Microsoft.ML.Tokenizers` gaps upstream

`LayaTokenizer` (`src/NLaya/Tokenization/LayaTokenizer.cs`) builds `BpeTokenizer` from
`tokenizer.json`, then fills three gaps for the Gemma-style multilingual vocabulary:

1. **Added tokens, longest-first**, with `lstrip` (for example `<mask>` swallows the space before it,
   and `\n\n` beats `\n`).
2. **Metaspace `prepend_scheme: always`**: a `▁` before each segment between added tokens.
3. **`byte_fallback`**: characters missing from the vocabulary become `<0xNN>` tokens.
   `BpeOptions` has no such option.

Check the latest `Microsoft.ML.Tokenizers` (3.0 previews were out in 2026) for a `tokenizer.json`
loader or these options. If they're missing, open an issue or PR on `dotnet/machinelearning`
with the fixture cases in `tests/NLaya.Tests/Fixtures/tokenizer_multilingual.json` as evidence.
When they land, `LayaTokenizer` shrinks to a load call plus the special-token lookup. The English
checkpoint (byte-level BPE) already matches with the library alone.

## 4d. Watch for a Microsoft safetensors loader

`src/NLaya.TorchSharp/SafeTensors.cs` (~70 lines) reads `model.safetensors` by hand.
`Microsoft.ML.GenAI.Core` has a loader, but it's internal. TorchSharp.PyBridge has one too, but
it's a community package, and the user prefers Microsoft ones. If either TorchSharp or GenAI.Core
makes a loader public, switch to it.

## Not worth removing

These look like candidates, but each is needed:
- `PythonJson`: byte-exact `json.dumps` is required for token parity.
- The ModernBERT forward pass (`LayaNetwork.cs`): no Microsoft package has the architecture.
- `PyStr`: Python string semantics the ported heuristics depend on.
