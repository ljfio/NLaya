using Microsoft.Extensions.Logging;

using NLaya.Backends;
using NLaya.Calibration;

namespace NLaya;

/// <summary>How to load a checkpoint. Backends add themselves via extension methods, e.g. <c>UseTorchSharp()</c>.</summary>
public sealed class LayaOptions
{
    /// <summary>The inference backend. Required: <c>o.UseTorchSharp()</c> or <c>o.UseOnnx(dir)</c>.</summary>
    public ILayaBackendFactory? Backend { get; set; }

    /// <summary>One checkpoint out of a repo that bundles several, e.g. "multilingual".</summary>
    public string? Subfolder { get; set; }

    /// <summary>The cached revision to load (a branch/tag name or commit hash).</summary>
    public string Revision { get; set; } = "main";

    /// <summary>Overrides the Hugging Face cache directory.</summary>
    public string? CacheDir { get; set; }

    /// <summary>Per-language calibration, keyed by language code ("de", "pt-BR" -> "pt").</summary>
    public IDictionary<string, LanguageTemperature> LangTemperatures { get; } = new Dictionary<string, LanguageTemperature>();

    public IList<ILayaHook> Hooks { get; } = new List<ILayaHook>();

    /// <summary>When false, a failing hook logs a warning and inference continues.</summary>
    public bool ThrowOnHookError { get; set; } = true;

    public ILogger? Logger { get; set; }
}
