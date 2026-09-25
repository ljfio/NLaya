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

    /// <summary>Run the encoder and decision head over <paramref name="batch"/>: option logits and act logits per row.</summary>
    BackendOutput Run(EncodedBatch batch);
}
