namespace NLaya.Hub;

/// <summary>
/// Finds checkpoints in the local Hugging Face cache. NLaya never downloads: fetch checkpoints with
/// the Hugging Face CLI (<c>hf download convaiinnovations/laya-multilingual</c>), which writes to
/// <c>$HF_HUB_CACHE</c>, else <c>$HF_HOME/hub</c>, else <c>~/.cache/huggingface/hub</c>. The Python
/// library uses the same cache, so its downloads are found too.
/// </summary>
public static class HfCache
{
    public static string DefaultDir()
    {
        if (Environment.GetEnvironmentVariable("HF_HUB_CACHE") is { Length: > 0 } hub) return hub;
        var home = Environment.GetEnvironmentVariable("HF_HOME");
        if (string.IsNullOrEmpty(home))
        {
            var xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            home = Path.Combine(string.IsNullOrEmpty(xdg)
                ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache")
                : xdg, "huggingface");
        }
        return Path.Combine(home, "hub");
    }

    /// <summary>The cached snapshot directory of <paramref name="repoId"/> at <paramref name="revision"/>, or null.</summary>
    public static string? Snapshot(string repoId, string revision = "main", string? cacheDir = null)
    {
        var repo = Path.Combine(cacheDir ?? DefaultDir(), "models--" + repoId.Replace("/", "--"));
        var refPath = Path.Combine(repo, "refs", revision);
        var commit = File.Exists(refPath) ? File.ReadAllText(refPath).Trim() : revision;
        var dir = Path.Combine(repo, "snapshots", commit);
        return Directory.Exists(dir) ? dir : null;
    }

    /// <summary>The CLI command that fetches a checkpoint.</summary>
    public static string DownloadCommand(string repoId, string? subfolder = null, string revision = "main") =>
        $"hf download {repoId}" + (subfolder is { Length: > 0 } s ? $" --include \"{s}/*\"" : "") +
        (revision != "main" ? $" --revision {revision}" : "");
}
