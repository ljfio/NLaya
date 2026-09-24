namespace NLaya.Backends;

/// <summary>
/// Runs the Laya network (encoder + decision head) on a collated batch. The core library does
/// tokenization, sequence building and decoding; a backend only turns tensors into logits.
/// Implementations must be safe to call from several threads at once.
/// </summary>
public interface ILayaBackend : IDisposable
{
    /// <summary>A short description, e.g. "torchsharp:cpu" or "onnx:cuda".</summary>
    string Name { get; }

    BackendOutput Run(EncodedBatch batch);
}
