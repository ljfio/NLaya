using NLaya.Backends;
using NLaya.Config;
using NLaya.Hub;
using NLaya.Tokenization;

namespace NLaya;

/// <summary>Entry point, like Python's <c>laya.load</c>.</summary>
public static class Laya
{
    /// <summary>The English checkpoint (and the bundle repo holding all three).</summary>
    public const string DefaultModel = "convaiinnovations/laya";
    public const string MultilingualModel = "convaiinnovations/laya-multilingual";

    internal static readonly string[] BaseFiles =
    [
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ];

    /// <summary>
    /// Load a checkpoint from a local directory or the Hugging Face Hub (downloaded into the HF
    /// cache on first use, then reused).
    /// <code>
    /// await using var agent = await Laya.LoadAsync("convaiinnovations/laya-multilingual", o => o.UseTorchSharp());
    /// </code>
    /// </summary>
    public static async Task<LayaAgent> LoadAsync(string modelIdOrPath = DefaultModel, Action<LayaOptions>? configure = null,
        CancellationToken ct = default)
    {
        var options = new LayaOptions();
        configure?.Invoke(options);
        var factory = options.Backend ?? throw new InvalidOperationException(
            "No inference backend configured. Add NLaya.TorchSharp and call o.UseTorchSharp(), " +
            "or add NLaya.Onnx and call o.UseOnnx(dir).");
        var dir = await ResolveAsync(modelIdOrPath, options, factory.RequiredFiles, ct).ConfigureAwait(false);
        var checkpoint = await Task.Run(() => ReadCheckpoint(modelIdOrPath, dir), ct).ConfigureAwait(false);
        var backend = await Task.Run(() => factory.Create(checkpoint), ct).ConfigureAwait(false);
        return new LayaAgent(checkpoint, backend, options);
    }

    /// <summary>Synchronous <see cref="LoadAsync"/>.</summary>
    public static LayaAgent Load(string modelIdOrPath = DefaultModel, Action<LayaOptions>? configure = null) =>
        LoadAsync(modelIdOrPath, configure).GetAwaiter().GetResult();

    internal static async Task<string> ResolveAsync(string modelIdOrPath, LayaOptions options, IReadOnlyList<string> extraFiles,
        CancellationToken ct)
    {
        string dir;
        if (System.IO.Directory.Exists(modelIdOrPath))
        {
            dir = modelIdOrPath;
        }
        else
        {
            if (Path.IsPathRooted(modelIdOrPath) || modelIdOrPath.StartsWith("./", StringComparison.Ordinal)
                || modelIdOrPath.StartsWith("../", StringComparison.Ordinal) || modelIdOrPath.Count(c => c == '/') != 1)
                throw new DirectoryNotFoundException(
                    $"Local model path not found: '{modelIdOrPath}'. Check that the directory exists and that training saved the model successfully.");
            var prefix = options.Subfolder is { Length: > 0 } s ? s.TrimEnd('/') + "/" : "";
            using var hub = new HfHubClient(options.Token, options.CacheDir);
            dir = await hub.DownloadAsync(modelIdOrPath, BaseFiles.Concat(extraFiles).Distinct().Select(f => prefix + f),
                options.Revision, options.DownloadProgress, ct).ConfigureAwait(false);
        }
        if (options.Subfolder is { Length: > 0 } sub)
        {
            dir = Path.Combine(dir, sub);
            if (!System.IO.Directory.Exists(dir))
                throw new DirectoryNotFoundException($"Subfolder '{sub}' not found in '{modelIdOrPath}'.");
        }
        return dir;
    }

    internal static LayaCheckpoint ReadCheckpoint(string modelId, string dir)
    {
        var cfgPath = Path.Combine(dir, "rl_agent_config.json");
        if (!File.Exists(cfgPath))
            throw new FileNotFoundException(
                $"Incompatible model: '{modelId}' does not contain 'rl_agent_config.json'. That file ships with the weights of a " +
                "Laya checkpoint, so load one of those (e.g. 'convaiinnovations/laya') or a directory your own training run wrote.", cfgPath);
        var cfg = AgentConfig.Load(cfgPath);
        var encPath = Path.Combine(dir, "encoder", "config.json");
        var enc = File.Exists(encPath) ? ModernBertConfig.Load(encPath) : null;
        var tokJson = FirstExisting(Path.Combine(dir, "tokenizer", "tokenizer.json"), Path.Combine(dir, "tokenizer.json"))
            ?? throw new FileNotFoundException($"'{modelId}' has no tokenizer/tokenizer.json.");
        var tokCfg = FirstExisting(Path.Combine(dir, "tokenizer", "tokenizer_config.json"), Path.Combine(dir, "tokenizer_config.json"));
        var tok = LayaTokenizer.FromFile(tokJson, tokCfg);
        return new LayaCheckpoint(modelId, dir, cfg, enc, tok);
    }

    private static string? FirstExisting(params string[] paths) => paths.FirstOrDefault(File.Exists);
}
