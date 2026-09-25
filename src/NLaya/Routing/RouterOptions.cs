using Microsoft.Extensions.Logging;

namespace NLaya.Routing;

/// <summary>Settings for a <see cref="Router"/>.</summary>
public sealed class RouterOptions
{
    /// <summary>Configure each checkpoint's load (backend, device, ...), given its name.</summary>
    public Action<Checkpoint, LayaOptions>? ConfigureAgent { get; set; }

    /// <summary>Override where a checkpoint lives.</summary>
    public IDictionary<Checkpoint, CheckpointSpec> Models { get; } = new Dictionary<Checkpoint, CheckpointSpec>();

    /// <summary>Load from the standalone repos instead of the <c>convaiinnovations/laya</c> bundle.</summary>
    public bool StandaloneRepos { get; set; }

    /// <summary>How many checkpoints stay resident (least recently used is dropped).</summary>
    public int MaxLoaded { get; set; } = 2;

    /// <summary>Checkpoint for text whose language cannot be identified.</summary>
    public Checkpoint Default { get; set; } = Checkpoint.English;

    /// <summary>Send questions whose ids match a typed-decisions workflow to that checkpoint.</summary>
    public bool AutoTaskDetection { get; set; }

    /// <summary>A language-identification hook: returns a code ("en", "pt-BR") or null to abstain.</summary>
    public Func<LayaState, string?>? LangGuess { get; set; }

    /// <summary>Hooks the router runs per request (the agents it loads don't run them again).</summary>
    public IList<ILayaHook> Hooks { get; } = new List<ILayaHook>();
    /// <summary>When false, a failing hook is logged and the call continues.</summary>
    public bool ThrowOnHookError { get; set; } = true;
    /// <summary>Where the router and the agents it loads log.</summary>
    public ILogger? Logger { get; set; }
}
