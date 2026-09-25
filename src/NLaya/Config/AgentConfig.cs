using System.Text.Json.Nodes;

namespace NLaya.Config;

/// <summary><c>rl_agent_config.json</c>: the decision head's shape, token budgets and calibration.</summary>
public sealed class AgentConfig
{
    /// <summary>The base encoder's Hub id, e.g. <c>answerdotai/ModernBERT-large</c>.</summary>
    public string? Encoder { get; init; }
    /// <summary>Transformer layers in the decision head.</summary>
    public int HeadLayers { get; init; } = 2;
    /// <summary>Default token budget per question row.</summary>
    public int MaxLen { get; init; } = 512;
    /// <summary>Default token budget for a question and its options.</summary>
    public int HeadMaxLen { get; init; } = 192;
    /// <summary>Number of act-head outputs: one per entry in <c>act_costs</c>, plus "act".</summary>
    public int ActOutputs { get; init; } = 2;
    /// <summary>The checkpoint's own name, e.g. <c>laya-typed-decisions</c>.</summary>
    public string? ModelName { get; init; }
    /// <summary>Per question type (choice, score, noul), as shipped (not yet clamped).</summary>
    public IReadOnlyList<JsonNode?> Temperature { get; init; } = [1.0, 1.0, 1.0];
    /// <summary>Per "type:bucket" key, as shipped (not yet clamped).</summary>
    public IReadOnlyDictionary<string, JsonNode?> TemperatureByOptions { get; init; } = new Dictionary<string, JsonNode?>();
    /// <summary>The whole file, for fields NLaya doesn't model.</summary>
    public JsonObject Raw { get; init; } = new();

    /// <summary>Read <paramref name="path"/>.</summary>
    public static AgentConfig Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parse <c>rl_agent_config.json</c> text.</summary>
    public static AgentConfig Parse(string json)
    {
        var o = JsonNode.Parse(json)!.AsObject();
        return new AgentConfig
        {
            Encoder = o["encoder"]?.GetValue<string>(),
            HeadLayers = o["head_layers"]?.GetValue<int>() ?? 2,
            MaxLen = o["max_len"]?.GetValue<int>() ?? 512,
            HeadMaxLen = o["head_max_len"]?.GetValue<int>() ?? 192,
            ActOutputs = (o["act_costs"] as JsonObject)?.Count + 1 ?? 1,
            ModelName = o["model_name"]?.GetValue<string>(),
            Temperature = (o["temperature"] as JsonArray)?.Select(n => n?.DeepClone()).ToList() ?? [1.0, 1.0, 1.0],
            TemperatureByOptions = (o["temperature_by_options"] as JsonObject)?
                .ToDictionary(kv => kv.Key, kv => kv.Value?.DeepClone()) ?? new Dictionary<string, JsonNode?>(),
            Raw = o,
        };
    }
}
