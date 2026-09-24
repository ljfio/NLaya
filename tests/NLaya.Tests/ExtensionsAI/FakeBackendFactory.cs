using NLaya.Backends;

namespace NLaya.Tests.ExtensionsAI;

/// <summary>A backend that needs no weights: every option gets the same logit.</summary>
internal sealed class FakeBackendFactory : ILayaBackendFactory
{
    public int Created;
    public List<FakeBackend> Backends { get; } = [];

    public IReadOnlyList<string> RequiredFiles => [];

    public ILayaBackend Create(LayaCheckpoint checkpoint)
    {
        Interlocked.Increment(ref Created);
        var b = new FakeBackend();
        lock (Backends) Backends.Add(b);
        return b;
    }

    internal sealed class FakeBackend : ILayaBackend
    {
        public bool Disposed { get; private set; }
        public string Name => "fake";

        public BackendOutput Run(EncodedBatch batch) =>
            new(new float[batch.Rows * batch.MaxMarkers], new float[batch.Rows * 2], batch.Rows, batch.MaxMarkers, 2);

        public void Dispose() => Disposed = true;
    }
}
