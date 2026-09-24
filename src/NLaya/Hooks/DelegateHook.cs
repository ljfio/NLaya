namespace NLaya;

/// <summary>A hook made from start/end delegates; see <see cref="LayaHooks.OnStart"/> and <see cref="LayaHooks.OnEnd"/>.</summary>
internal sealed class DelegateHook(Action<PredictContext>? start, Action<PredictContext>? end) : ILayaHook
{
    public void OnPredictStart(PredictContext ctx) => start?.Invoke(ctx);
    public void OnPredictEnd(PredictContext ctx) => end?.Invoke(ctx);
}
