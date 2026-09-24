using System.Text.Json.Nodes;

namespace NLaya.Config;

/// <summary><c>rl_agent_config.json</c>: the decision head's shape, token budgets and calibration.</summary>
public sealed class AgentConfig
{
    public string? Encoder { get; init; }
    public int HeadLayers { get; init; } = 2;
    public int MaxLen { get; init; } = 512;
    public int HeadMaxLen { get; init; } = 192;
    /// <summary>Number of act-head outputs: one per entry in <c>act_costs</c>, plus "act".</summary>
    public int ActOutputs { get; init; } = 2;
    public string? ModelName { get; init; }
    /// <summary>Per question type (choice, score, noul), as shipped (not yet clamped).</summary>
    public IReadOnlyList<JsonNode?> Temperature { get; init; } = [1.0, 1.0, 1.0];
    /// <summary>Per "type:bucket" key, as shipped (not yet clamped).</summary>
    public IReadOnlyDictionary<string, JsonNode?> TemperatureByOptions { get; init; } = new Dictionary<string, JsonNode?>();
    public JsonObject Raw { get; init; } = new();

    public static AgentConfig Load(string path) => Parse(File.ReadAllText(path));

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
