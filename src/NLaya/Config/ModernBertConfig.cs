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
    /// <summary>Width of the hidden states.</summary>
    public int HiddenSize { get; init; } = 768;
    /// <summary>Encoder layers.</summary>
    public int NumHiddenLayers { get; init; } = 22;
    /// <summary>Attention heads per layer.</summary>
    public int NumAttentionHeads { get; init; } = 12;
    /// <summary>Width of the MLP (per GLU half).</summary>
    public int IntermediateSize { get; init; } = 1152;
    /// <summary>Token embeddings.</summary>
    public int VocabSize { get; init; } = 50368;
    /// <summary>Longest sequence the rotary embeddings were trained for.</summary>
    public int MaxPositionEmbeddings { get; init; } = 8192;
    /// <summary>LayerNorm epsilon.</summary>
    public double NormEps { get; init; } = 1e-5;
    /// <summary>Whether LayerNorms have a bias.</summary>
    public bool NormBias { get; init; }
    /// <summary>Whether the attention projections have a bias.</summary>
    public bool AttentionBias { get; init; }
    /// <summary>Whether the MLP projections have a bias.</summary>
    public bool MlpBias { get; init; }
    /// <summary>Every n-th layer attends globally; the rest use a sliding window (without <c>layer_types</c>).</summary>
    public int GlobalAttnEveryNLayers { get; init; } = 3;
    /// <summary>Sliding-window width in tokens (each side sees half).</summary>
    public int LocalAttention { get; init; } = 128;
    /// <summary>RoPE base for global layers.</summary>
    public double GlobalRopeTheta { get; init; } = 160000;
    /// <summary>RoPE base for sliding-window layers.</summary>
    public double LocalRopeTheta { get; init; } = 10000;
    /// <summary>The MLP activation, e.g. <c>gelu</c>.</summary>
    public string HiddenActivation { get; init; } = "gelu";
    /// <summary>The padding token id.</summary>
    public int PadTokenId { get; init; }
    /// <summary>Per layer: true for full (global) attention, false for sliding-window.</summary>
    public IReadOnlyList<bool> GlobalLayers { get; init; } = [];

    /// <summary>Width of one attention head.</summary>
    public int HeadDim => HiddenSize / NumAttentionHeads;

    /// <summary>Read <paramref name="path"/>.</summary>
    public static ModernBertConfig Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parse <c>config.json</c> text.</summary>
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
