using System.Text.Json;
using System.Text.Json.Nodes;

namespace NLaya.Calibration;

/// <summary>
/// Temperatures from <see cref="TemperatureFitter"/>: per question type, and optionally per
/// option-count bucket. Save them into a checkpoint's <c>rl_agent_config.json</c>
/// (<see cref="SaveConfig"/>), or apply them at load time for one language with
/// <see cref="ToLanguageTemperature"/> and <see cref="LayaOptions.LangTemperatures"/>.
/// </summary>
public sealed class TemperatureFit
{
    internal TemperatureFit(double[] perType, Dictionary<string, double> byOptions, Dictionary<string, int> sampleCounts)
    {
        PerType = perType;
        ByOptions = byOptions;
        SampleCounts = sampleCounts;
    }

    /// <summary>Temperatures for choice, score and noul questions, in that order (the config's <c>temperature</c>).</summary>
    public IReadOnlyList<double> PerType { get; }

    /// <summary>Per-bucket temperatures, e.g. "choice:3-5" (the config's <c>temperature_by_options</c>); empty for a per-type fit.</summary>
    public IReadOnlyDictionary<string, double> ByOptions { get; }

    /// <summary>Samples seen per type name or bucket, including buckets too small to fit.</summary>
    public IReadOnlyDictionary<string, int> SampleCounts { get; }

    /// <summary>
    /// A copy of <paramref name="config"/> (an <c>rl_agent_config.json</c> object) with these
    /// temperatures. Like the fine-tuning notebook's export, a fit without buckets removes
    /// <c>temperature_by_options</c>, since inherited buckets take precedence over per-type values.
    /// </summary>
    public JsonObject ApplyTo(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        var o = config.DeepClone().AsObject();
        o["temperature"] = new JsonArray(PerType.Select(t => (JsonNode?)JsonValue.Create(t)).ToArray());
        if (ByOptions.Count == 0) o.Remove("temperature_by_options");
        else o["temperature_by_options"] = new JsonObject(ByOptions.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)JsonValue.Create(kv.Value))));
        return o;
    }

    /// <summary>
    /// Read <paramref name="sourcePath"/>, apply the temperatures and write <paramref name="destinationPath"/>.
    /// Write into your own copy of the checkpoint: files in the Hugging Face cache are links to shared
    /// blobs, so writing to one is refused.
    /// </summary>
    public void SaveConfig(string sourcePath, string destinationPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(sourcePath);
        ArgumentException.ThrowIfNullOrEmpty(destinationPath);
        if (File.Exists(destinationPath) && new FileInfo(destinationPath).LinkTarget is not null)
            throw new IOException($"'{destinationPath}' is a link (a Hugging Face cache file?); copy the checkpoint and write the copy's config instead.");
        var config = JsonNode.Parse(File.ReadAllText(sourcePath))?.AsObject()
            ?? throw new InvalidDataException($"'{sourcePath}' is not a JSON object");
        File.WriteAllText(destinationPath, ApplyTo(config).ToJsonString(Indented));
    }

    /// <summary>These temperatures as a per-language override, for a language calibrated on its own data.</summary>
    public LanguageTemperature ToLanguageTemperature() => new()
    {
        Temperature = PerType.ToArray(),
        TemperatureByOptions = ByOptions.Count == 0 ? null : new Dictionary<string, double>(ByOptions, StringComparer.Ordinal),
    };

    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };
}
