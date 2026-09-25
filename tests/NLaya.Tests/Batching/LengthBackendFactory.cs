using NLaya.Backends;

namespace NLaya.Tests.Batching;

/// <summary>
/// A backend that needs no weights but still tells states apart: each row's winning option is its
/// token length modulo its option count, so a result that lands on the wrong state shows up.
/// </summary>
internal sealed class LengthBackendFactory : ILayaBackendFactory
{
    public IReadOnlyList<string> RequiredFiles => [];

    public ILayaBackend Create(LayaCheckpoint checkpoint) => new LengthBackend();

    private sealed class LengthBackend : ILayaBackend
    {
        public string Name => "length";

        public BackendOutput Run(EncodedBatch batch)
        {
            var logits = new float[batch.Rows * batch.MaxMarkers];
            for (var r = 0; r < batch.Rows; r++)
            {
                var k = batch.OptionCounts[r];
                for (var m = 0; m < batch.MaxMarkers; m++)
                    logits[r * batch.MaxMarkers + m] = m >= k ? -1e4f : m == batch.Lengths[r] % k ? 2f : 0f;
            }
            return new BackendOutput(logits, new float[batch.Rows * 2], batch.Rows, batch.MaxMarkers, 2);
        }

        public void Dispose() { }
    }
}
