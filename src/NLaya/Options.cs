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
    public bool HooksRaise { get; set; } = true;

    public ILogger? Logger { get; set; }
}

/// <summary>Per-call settings for <see cref="LayaAgent.Predict(LayaState, Questions, PredictOptions?)"/>.</summary>
public class PredictOptions
{
    /// <summary>Language code; selects a <see cref="LayaOptions.LangTemperatures"/> override.</summary>
    public string? Lang { get; init; }

    /// <summary>Token budget per question row. <c>laya-multilingual</c> reads up to 8192.</summary>
    public int? MaxLen { get; init; }

    /// <summary>Token budget for the question and its options.</summary>
    public int? HeadMaxLen { get; init; }

    /// <summary>Hooks for this call only, run after installed hooks.</summary>
    public IEnumerable<ILayaHook>? Hooks { get; init; }

    public bool? HooksRaise { get; init; }
}

/// <summary>Per-call settings for <see cref="LayaAgent.PredictBatch"/>.</summary>
public sealed class BatchOptions : PredictOptions
{
    /// <summary>States per forward pass; null sends them all in one.</summary>
    public int? BatchSize { get; init; }

    /// <summary>
    /// Group similar-length states (within windows of eight batches) to reduce padding. Needs a
    /// <see cref="BatchSize"/> between 1 and the number of states. Results keep input order.
    /// </summary>
    public bool SortByLength { get; init; }
}
