namespace NLaya.Tests.Batching;

/// <summary>Records every predict event it sees, and can act in start hooks.</summary>
internal sealed class RecordingHook(Action<PredictContext>? onStart = null) : ILayaHook
{
    public List<string> Events { get; } = [];

    public void OnPredictStart(PredictContext ctx)
    {
        Events.Add("start:" + ctx.States[0].Serialize());
        onStart?.Invoke(ctx);
    }

    public void OnPredictEnd(PredictContext ctx) => Events.Add("end:" + ctx.States[0].Serialize());

    public void OnError(PredictContext ctx) => Events.Add("error:" + ctx.States[0].Serialize());
}
