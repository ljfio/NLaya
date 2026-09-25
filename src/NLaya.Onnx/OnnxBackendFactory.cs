using NLaya.Backends;

namespace NLaya.Onnx;

/// <summary>Creates <see cref="OnnxBackend"/>s from an <c>export_onnx.py</c> output directory; <c>UseOnnx</c> sets it up.</summary>
public sealed class OnnxBackendFactory(OnnxOptions options) : ILayaBackendFactory
{
    /// <summary>None from the checkpoint: the ONNX files live in <see cref="OnnxOptions.ModelDir"/>.</summary>
    public IReadOnlyList<string> RequiredFiles { get; } = [];

    /// <inheritdoc/>
    public ILayaBackend Create(LayaCheckpoint checkpoint) => new OnnxBackend(options);
}
