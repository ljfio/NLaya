using NLaya.Config;
using NLaya.Tokenization;

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

/// <summary>Everything a backend needs to build itself from a checkpoint directory.</summary>
public sealed record LayaCheckpoint(
    string ModelId,
    string Directory,
    AgentConfig Config,
    ModernBertConfig? EncoderConfig,
    LayaTokenizer Tokenizer)
{
    public string WeightsPath => Path.Combine(Directory, "model.safetensors");
}

/// <summary>Creates a backend for a checkpoint, and says which checkpoint files it needs.</summary>
public interface ILayaBackendFactory
{
    /// <summary>Checkpoint files the backend needs besides config and tokenizer (e.g. "model.safetensors").</summary>
    IReadOnlyList<string> RequiredFiles { get; }

    ILayaBackend Create(LayaCheckpoint checkpoint);
}
