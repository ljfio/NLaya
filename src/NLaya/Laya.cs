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

    /// <summary>
    /// Load a checkpoint from a local directory, or by Hub id from the Hugging Face cache. NLaya does
    /// not download: fetch checkpoints first with <c>hf download &lt;repo&gt;</c>.
    /// <code>
    /// await using var agent = await Laya.LoadAsync("convaiinnovations/laya-multilingual", o => o.UseTorchSharp());
    /// </code>
    /// </summary>
    public static Task<LayaAgent> LoadAsync(string modelIdOrPath = DefaultModel, Action<LayaOptions>? configure = null,
        CancellationToken ct = default) =>
        Task.Run(() => LayaTelemetry.Load(modelIdOrPath, () => LoadCore(modelIdOrPath, configure, ct)), ct);

    /// <summary>Synchronous <see cref="LoadAsync"/>: reads the checkpoint and builds the backend on the calling thread.</summary>
    public static LayaAgent Load(string modelIdOrPath = DefaultModel, Action<LayaOptions>? configure = null) =>
        LayaTelemetry.Load(modelIdOrPath, () => LoadCore(modelIdOrPath, configure, CancellationToken.None));

    private static LayaAgent LoadCore(string modelIdOrPath, Action<LayaOptions>? configure, CancellationToken ct)
    {
        var options = new LayaOptions();
        configure?.Invoke(options);
        var factory = options.Backend ?? throw new InvalidOperationException(
            "No inference backend configured. Add NLaya.TorchSharp and call o.UseTorchSharp(), " +
            "or add NLaya.Onnx and call o.UseOnnx(dir).");
        var dir = Resolve(modelIdOrPath, options, factory.RequiredFiles);
        var checkpoint = ReadCheckpoint(modelIdOrPath, dir);
        ct.ThrowIfCancellationRequested(); // before the expensive part: building the backend loads the weights
        return new LayaAgent(checkpoint, factory.Create(checkpoint), options);
    }

    /// <summary>A local directory, or a Hub id looked up in the Hugging Face cache (see <see cref="HfCache"/>).</summary>
    internal static string Resolve(string modelIdOrPath, LayaOptions options, IReadOnlyList<string> requiredFiles)
    {
        var sub = options.Subfolder is { Length: > 0 } s ? s.Trim('/') : null;
        string dir;
        if (System.IO.Directory.Exists(modelIdOrPath))
        {
            dir = modelIdOrPath;
        }
        else
        {
            if (Path.IsPathRooted(modelIdOrPath) || modelIdOrPath.StartsWith('.') || modelIdOrPath.Count(c => c == '/') != 1)
                throw new DirectoryNotFoundException($"Local model path not found: '{modelIdOrPath}'.");
            dir = HfCache.Snapshot(modelIdOrPath, options.Revision, options.CacheDir) ?? throw new DirectoryNotFoundException(
                $"'{modelIdOrPath}' is not in the Hugging Face cache ({options.CacheDir ?? HfCache.DefaultDir()}). " +
                $"Download it first: {HfCache.DownloadCommand(modelIdOrPath, sub, options.Revision)}");
        }
        if (sub is not null)
        {
            dir = Path.Combine(dir, sub);
            if (!System.IO.Directory.Exists(dir))
                throw new DirectoryNotFoundException($"Subfolder '{sub}' not found in '{modelIdOrPath}'. " +
                    $"Download it with: {HfCache.DownloadCommand(modelIdOrPath, sub, options.Revision)}");
        }
        var missing = requiredFiles.Where(f => !File.Exists(Path.Combine(dir, f))).ToList();
        if (missing.Count > 0)
            throw new FileNotFoundException($"'{modelIdOrPath}' is missing {string.Join(", ", missing)} (in {dir})." +
                (System.IO.Directory.Exists(modelIdOrPath) ? "" : $" Download it with: {HfCache.DownloadCommand(modelIdOrPath, sub, options.Revision)}"));
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
