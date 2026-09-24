using NLaya.Backends;

namespace NLaya.Onnx;

public sealed class OnnxBackendFactory(OnnxOptions options) : ILayaBackendFactory
{
    public IReadOnlyList<string> RequiredFiles { get; } = [];

    public ILayaBackend Create(LayaCheckpoint checkpoint) => new OnnxBackend(options);
}
