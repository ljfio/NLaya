namespace NLaya.Backends;

/// <summary>Creates a backend for a checkpoint, and says which checkpoint files it needs.</summary>
public interface ILayaBackendFactory
{
    /// <summary>Checkpoint files the backend needs besides config and tokenizer (e.g. "model.safetensors").</summary>
    IReadOnlyList<string> RequiredFiles { get; }

    /// <summary>Build a backend (load weights) for <paramref name="checkpoint"/>.</summary>
    ILayaBackend Create(LayaCheckpoint checkpoint);
}
