using NLaya.Backends;

namespace NLaya.TorchSharp;

/// <summary>Runs Laya with TorchSharp (libtorch), loading <c>model.safetensors</c> directly.</summary>
public sealed class TorchSharpBackendFactory(TorchSharpOptions options) : ILayaBackendFactory
{
    /// <inheritdoc/>
    public IReadOnlyList<string> RequiredFiles { get; } = ["model.safetensors"];

    /// <inheritdoc/>
    public ILayaBackend Create(LayaCheckpoint checkpoint) => new TorchSharpBackend(checkpoint, options);
}
