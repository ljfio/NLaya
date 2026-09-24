namespace NLaya.Calibration;

/// <summary>A per-language temperature override (Python <c>lang_temperatures</c> entry).</summary>
public sealed class LanguageTemperature
{
    /// <summary>Per question type (choice, score, noul). Null inherits the checkpoint's.</summary>
    public IReadOnlyList<double>? Temperature { get; init; }

    /// <summary>Per "type:bucket" key, e.g. "choice:11+". Only these; the checkpoint's buckets are not inherited.</summary>
    public IReadOnlyDictionary<string, double>? TemperatureByOptions { get; init; }
}
