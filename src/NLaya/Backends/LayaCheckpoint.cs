using NLaya.Config;
using NLaya.Tokenization;

namespace NLaya.Backends;

/// <summary>Everything a backend needs to build itself from a checkpoint directory.</summary>
public sealed record LayaCheckpoint(
    string ModelId,
    string Directory,
    AgentConfig Config,
    ModernBertConfig? EncoderConfig,
    LayaTokenizer Tokenizer)
{
    public string WeightsPath => Path.Combine(Directory, "model.safetensors");
}
