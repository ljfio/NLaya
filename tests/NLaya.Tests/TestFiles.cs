using System.Text.Json.Nodes;

using NLaya.Hub;
using NLaya.Tokenization;

namespace NLaya.Tests;

/// <summary>Fixture JSON (committed) and checkpoint files (from the Hugging Face cache).</summary>
internal static class TestFiles
{
    public static JsonNode Fixture(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)))!;

    private static readonly Dictionary<string, LayaTokenizer> Tokenizers = new();

    /// <summary>A checkpoint's tokenizer from the Hugging Face cache, or a skipped test when it is not there.</summary>
    public static LayaTokenizer Tokenizer(string repo)
    {
        lock (Tokenizers)
        {
            if (Tokenizers.TryGetValue(repo, out var t)) return t;
            var dir = HfCache.Snapshot(repo);
            if (dir is null || !File.Exists(Path.Combine(dir, "tokenizer", "tokenizer.json")))
                Assert.Skip($"tokenizer for {repo} is not cached; run: {HfCache.DownloadCommand(repo)}");
            t = LayaTokenizer.FromFile(Path.Combine(dir, "tokenizer", "tokenizer.json"), Path.Combine(dir, "tokenizer", "tokenizer_config.json"));
            Tokenizers[repo] = t;
            return t;
        }
    }
}
