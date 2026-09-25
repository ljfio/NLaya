namespace NLaya;

/// <summary>
/// Opt-in lifecycle hooks (port of <c>laya.hooks</c>). Implement only the events you need; the rest
/// default to no-ops. Start hooks may rewrite <see cref="PredictContext.States"/> /
/// <see cref="PredictContext.Questions"/> or call <see cref="PredictContext.Skip"/>; end hooks may
/// rewrite <see cref="PredictContext.Results"/>.
/// </summary>
public interface ILayaHook
{
    /// <summary>Before inference: may rewrite the states or questions, or call <see cref="PredictContext.Skip"/>.</summary>
    void OnPredictStart(PredictContext ctx) { }
    /// <summary>After inference (or a failure): may rewrite <see cref="PredictContext.Results"/>.</summary>
    void OnPredictEnd(PredictContext ctx) { }
    /// <summary>Router: may replace <see cref="PredictContext.Decision"/>.</summary>
    void OnRoute(PredictContext ctx) { }
    /// <summary>Router: checkpoint <see cref="PredictContext.Model"/> was loaded.</summary>
    void OnLoad(PredictContext ctx) { }
    /// <summary>Router: checkpoint <see cref="PredictContext.Model"/> was evicted or unloaded.</summary>
    void OnEvict(PredictContext ctx) { }
    /// <summary>Inference failed with <see cref="PredictContext.Error"/>; end hooks still run.</summary>
    void OnError(PredictContext ctx) { }
}
