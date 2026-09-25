namespace NLaya.Routing;

/// <summary>
/// One request of a heterogeneous <see cref="Router.PredictBatch(IEnumerable{RouteRequest}, BatchOptions?)"/>
/// or <see cref="Router.RouteBatch"/> (Python's request dict): its own state, questions and routing overrides.
/// </summary>
public sealed record RouteRequest(LayaState State, Questions Questions)
{
    /// <summary>Pin a checkpoint by name or alias; see <see cref="RouteOptions.Model"/>.</summary>
    public string? Model { get; init; }

    /// <summary>"typed_decisions" picks the typed-decisions checkpoint; see <see cref="RouteOptions.Task"/>.</summary>
    public string? Task { get; init; }

    /// <summary>The request's language code: routes, and selects per-language temperatures.</summary>
    public string? Lang { get; init; }

    /// <summary>A language-identification hint for this request; see <see cref="RouterOptions.LangGuess"/>.</summary>
    public Func<LayaState, string?>? LangGuess { get; init; }

    internal RouteOptions ToRouteOptions(IEnumerable<ILayaHook>? hooks, bool? hooksRaise) => new()
    {
        Model = Model,
        Task = Task,
        Lang = Lang,
        LangGuess = LangGuess,
        Hooks = hooks,
        HooksRaise = hooksRaise,
    };
}
