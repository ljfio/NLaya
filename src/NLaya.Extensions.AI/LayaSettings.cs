namespace NLaya.Extensions.AI;

/// <summary>
/// The parts of NLaya's setup that are configuration rather than code, bound from the <c>"Laya"</c>
/// section (<c>"Laya:&lt;key&gt;"</c> for a keyed agent). The backend stays in code, since each backend
/// is its own package. Values set in code (the <c>model</c> argument, the configure callbacks) win.
/// </summary>
public sealed class LayaSettings
{
    public const string SectionName = "Laya";

    /// <summary>Hub id or local path; <see cref="Laya.DefaultModel"/> when unset.</summary>
    public string? Model { get; set; }

    /// <summary>See <see cref="LayaOptions.Subfolder"/>.</summary>
    public string? Subfolder { get; set; }

    /// <summary>See <see cref="LayaOptions.Revision"/>. Also applies to every checkpoint a Router loads.</summary>
    public string? Revision { get; set; }

    /// <summary>See <see cref="LayaOptions.CacheDir"/>. Also applies to every checkpoint a Router loads.</summary>
    public string? CacheDir { get; set; }

    /// <summary>Load the model while the host starts, so the first request doesn't pay for it. On by default.</summary>
    public bool Warmup { get; set; } = true;

    /// <summary>Router: see <see cref="Routing.RouterOptions.MaxLoaded"/>.</summary>
    public int? MaxLoaded { get; set; }

    /// <summary>Router: see <see cref="Routing.RouterOptions.Default"/>.</summary>
    public string? Default { get; set; }

    /// <summary>Router: see <see cref="Routing.RouterOptions.AutoTaskDetection"/>.</summary>
    public bool? AutoTaskDetection { get; set; }

    /// <summary>Router: see <see cref="Routing.RouterOptions.StandaloneRepos"/>.</summary>
    public bool? StandaloneRepos { get; set; }

    /// <summary>Router: the checkpoints to load at warm-up. Just the router's default checkpoint when unset.</summary>
    public string[]? Preload { get; set; }
}
