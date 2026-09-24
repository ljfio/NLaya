using System.Text.Json.Nodes;
using NLaya.Hub;
using NLaya.Tokenization;

namespace NLaya.Tests;

/// <summary>Fixture JSON (committed) and Hub files (from the HF cache, downloaded if missing).</summary>
internal static class TestFiles
{
    public static JsonNode Fixture(string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)))!;

    private static readonly Dictionary<string, LayaTokenizer> Tokenizers = new();

    /// <summary>A checkpoint's tokenizer, or a skipped test when the Hub is unreachable and nothing is cached.</summary>
    public static LayaTokenizer Tokenizer(string repo)
    {
        lock (Tokenizers)
        {
            if (Tokenizers.TryGetValue(repo, out var t)) return t;
            string dir;
            try
            {
                using var hub = new HfHubClient();
                dir = hub.DownloadAsync(repo, ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]).GetAwaiter().GetResult();
            }
            catch (Exception e) when (e is InvalidOperationException or HttpRequestException)
            {
                Assert.Skip($"tokenizer for {repo} unavailable: {e.Message}");
                throw;
            }
            t = LayaTokenizer.FromFile(Path.Combine(dir, "tokenizer", "tokenizer.json"), Path.Combine(dir, "tokenizer", "tokenizer_config.json"));
            Tokenizers[repo] = t;
            return t;
        }
    }
}
