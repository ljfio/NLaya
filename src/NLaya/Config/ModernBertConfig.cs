using System.Text.Json.Nodes;

namespace NLaya.Config;

/// <summary>
/// <c>encoder/config.json</c> for the ModernBERT family (ModernBERT-large for the English
/// checkpoints, mmBERT-base for multilingual). Reads both the transformers 4.x fields
/// (<c>global_rope_theta</c>, <c>local_rope_theta</c>) and the 5.x <c>rope_parameters</c> /
/// <c>layer_types</c> shape.
/// </summary>
public sealed class ModernBertConfig
{
    public int HiddenSize { get; init; } = 768;
    public int NumHiddenLayers { get; init; } = 22;
    public int NumAttentionHeads { get; init; } = 12;
    public int IntermediateSize { get; init; } = 1152;
    public int VocabSize { get; init; } = 50368;
    public int MaxPositionEmbeddings { get; init; } = 8192;
    public double NormEps { get; init; } = 1e-5;
    public bool NormBias { get; init; }
    public bool AttentionBias { get; init; }
    public bool MlpBias { get; init; }
    public int GlobalAttnEveryNLayers { get; init; } = 3;
    public int LocalAttention { get; init; } = 128;
    public double GlobalRopeTheta { get; init; } = 160000;
    public double LocalRopeTheta { get; init; } = 10000;
    public string HiddenActivation { get; init; } = "gelu";
    public int PadTokenId { get; init; }
    /// <summary>Per layer: true for full (global) attention, false for sliding-window.</summary>
    public IReadOnlyList<bool> GlobalLayers { get; init; } = [];

    public int HeadDim => HiddenSize / NumAttentionHeads;

    public static ModernBertConfig Load(string path) => Parse(File.ReadAllText(path));

    public static ModernBertConfig Parse(string json)
    {
        var o = JsonNode.Parse(json)!.AsObject();
        int I(string k, int d) => o[k] is JsonValue v ? (int)v.GetValue<double>() : d;
        double D(string k, double d) => o[k] is JsonValue v ? v.GetValue<double>() : d;
        bool B(string k, bool d) => o[k] is JsonValue v ? v.GetValue<bool>() : d;

        var layers = I("num_hidden_layers", 22);
        var every = I("global_attn_every_n_layers", 3);
        var globalTheta = D("global_rope_theta", 160000);
        var localTheta = D("local_rope_theta", 10000);
        if (o["rope_parameters"] is JsonObject rope)
        {
            var flat = rope["rope_theta"] is JsonValue f ? f.GetValue<double>() : (double?)null;
            globalTheta = (rope["full_attention"] as JsonObject)?["rope_theta"]?.GetValue<double>() ?? flat ?? globalTheta;
            localTheta = (rope["sliding_attention"] as JsonObject)?["rope_theta"]?.GetValue<double>() ?? flat ?? localTheta;
        }
        else if (o["rope_theta"] is JsonValue rt)
        {
            globalTheta = rt.GetValue<double>();
        }

        List<bool> globalLayers;
        if (o["layer_types"] is JsonArray types && types.Count == layers)
            globalLayers = types.Select(t => t!.GetValue<string>() != "sliding_attention").ToList();
        else
            globalLayers = Enumerable.Range(0, layers).Select(i => i % every == 0).ToList();

        return new ModernBertConfig
        {
            HiddenSize = I("hidden_size", 768),
            NumHiddenLayers = layers,
            NumAttentionHeads = I("num_attention_heads", 12),
            IntermediateSize = I("intermediate_size", 1152),
            VocabSize = I("vocab_size", 50368),
            MaxPositionEmbeddings = I("max_position_embeddings", 8192),
            NormEps = D("norm_eps", D("layer_norm_eps", 1e-5)),
            NormBias = B("norm_bias", false),
            AttentionBias = B("attention_bias", false),
            MlpBias = B("mlp_bias", false),
            GlobalAttnEveryNLayers = every,
            LocalAttention = I("local_attention", 128),
            GlobalRopeTheta = globalTheta,
            LocalRopeTheta = localTheta,
            HiddenActivation = o["hidden_activation"]?.GetValue<string>() ?? "gelu",
            PadTokenId = I("pad_token_id", 0),
            GlobalLayers = globalLayers,
        };
    }
}
