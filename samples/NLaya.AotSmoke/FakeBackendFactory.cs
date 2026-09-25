using NLaya.Backends;

namespace NLaya.AotSmoke;

/// <summary>A backend that needs no weights: every option gets the same logit.</summary>
internal sealed class FakeBackendFactory : ILayaBackendFactory
{
    public IReadOnlyList<string> RequiredFiles => [];

    public ILayaBackend Create(LayaCheckpoint checkpoint) => new FakeBackend();

    private sealed class FakeBackend : ILayaBackend
    {
        public string Name => "fake";

        public BackendOutput Run(EncodedBatch batch) =>
            new(new float[batch.Rows * batch.MaxMarkers], new float[batch.Rows * 2], batch.Rows, batch.MaxMarkers, 2);

        public void Dispose() { }
    }
}
