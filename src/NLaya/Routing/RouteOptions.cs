namespace NLaya.Routing;

/// <summary>Per-call routing (and prediction) settings; precedence is Model, Task, Lang, LangGuess, then detection.</summary>
public sealed class RouteOptions : PredictOptions
{
    /// <summary>Pin a checkpoint by name or alias ("en", "ml", "typed").</summary>
    public string? Model { get; init; }

    /// <summary>"typed_decisions" picks the typed-decisions checkpoint.</summary>
    public string? Task { get; init; }

    /// <summary>A per-call language-identification hint; see <see cref="RouterOptions.LangGuess"/>.</summary>
    public Func<LayaState, string?>? LangGuess { get; init; }
}
