using Microsoft.Extensions.Logging;

namespace NLaya.Routing;

/// <summary>Settings for a <see cref="Router"/>.</summary>
public sealed class RouterOptions
{
    /// <summary>Configure each checkpoint's load (backend, device, ...), given its name.</summary>
    public Action<string, LayaOptions>? ConfigureAgent { get; set; }

    /// <summary>Override where a checkpoint lives, by name ("english", "multilingual", "typed-decisions").</summary>
    public IDictionary<string, CheckpointSpec> Models { get; } = new Dictionary<string, CheckpointSpec>();

    /// <summary>Load from the standalone repos instead of the <c>convaiinnovations/laya</c> bundle.</summary>
    public bool StandaloneRepos { get; set; }

    /// <summary>How many checkpoints stay resident (least recently used is dropped).</summary>
    public int MaxLoaded { get; set; } = 2;

    /// <summary>Checkpoint for text whose language cannot be identified.</summary>
    public string Default { get; set; } = "english";

    /// <summary>Send questions whose ids match a typed-decisions workflow to that checkpoint.</summary>
    public bool AutoTaskDetection { get; set; }

    /// <summary>A language-identification hook: returns a code ("en", "pt-BR") or null to abstain.</summary>
    public Func<LayaState, string?>? LangGuess { get; set; }

    public IList<ILayaHook> Hooks { get; } = new List<ILayaHook>();
    public bool HooksRaise { get; set; } = true;
    public ILogger? Logger { get; set; }
}
