namespace NLaya.Routing;

/// <summary>Per-call routing (and prediction) settings; precedence is Checkpoint, Lang, LangGuess, then detection.</summary>
public sealed class RouteOptions : PredictOptions
{
    /// <summary>Pin a checkpoint, skipping detection. Python's <c>model=</c> and <c>task="typed_decisions"</c>.</summary>
    public Checkpoint? Checkpoint { get; init; }


    /// <summary>A per-call language-identification hint; see <see cref="RouterOptions.LangGuess"/>.</summary>
    public Func<LayaState, string?>? LangGuess { get; init; }
}
