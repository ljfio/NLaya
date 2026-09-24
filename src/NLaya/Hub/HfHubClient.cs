using System.Net;
using System.Net.Http.Headers;

namespace NLaya.Hub;

/// <summary>
/// Downloads files from the Hugging Face Hub into the standard HF cache
/// (<c>$HF_HUB_CACHE</c>, else <c>$HF_HOME/hub</c>, else <c>~/.cache/huggingface/hub</c>), laid out as
/// <c>models--org--name/snapshots/&lt;commit&gt;/path</c>, so a checkpoint the Python library has
/// already downloaded is reused and vice versa. Honours <c>HF_TOKEN</c>, <c>HF_ENDPOINT</c> and
/// <c>HF_HUB_OFFLINE=1</c>.
/// </summary>
public sealed class HfHubClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly HttpClient _noRedirect;
    private readonly string? _token;
    private readonly string _endpoint;

    public string CacheDir { get; }
    public bool Offline { get; }

    public HfHubClient(string? token = null, string? cacheDir = null, HttpMessageHandler? handler = null)
    {
        _token = token ?? Environment.GetEnvironmentVariable("HF_TOKEN");
        _endpoint = (Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? "https://huggingface.co").TrimEnd('/');
        CacheDir = cacheDir ?? DefaultCacheDir();
        Offline = Environment.GetEnvironmentVariable("HF_HUB_OFFLINE") is "1" or "true" or "TRUE";
        _http = handler is null ? new HttpClient() : new HttpClient(handler, disposeHandler: false);
        _noRedirect = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false });
        _http.Timeout = Timeout.InfiniteTimeSpan;
    }

    public static string DefaultCacheDir()
    {
        var hub = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrEmpty(hub)) return hub;
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

    private string RepoDir(string repoId) => Path.Combine(CacheDir, "models--" + repoId.Replace("/", "--"));

    /// <summary>
    /// Ensure <paramref name="files"/> (repo-relative paths) are in the cache and return the
    /// snapshot directory holding them. Files already cached for the resolved commit are not
    /// downloaded again.
    /// </summary>
    public async Task<string> DownloadAsync(string repoId, IEnumerable<string> files, string revision = "main",
        IProgress<(string File, long Done, long? Total)>? progress = null, CancellationToken ct = default)
    {
        var repoDir = RepoDir(repoId);
        var fileList = files.ToList();
        var commit = Offline ? null : await TryResolveCommitAsync(repoId, fileList[0], revision, ct).ConfigureAwait(false);
        if (commit is null)
        {
            // Offline, or the Hub is unreachable: fall back to the last snapshot we know of.
            var refPath = Path.Combine(repoDir, "refs", revision);
            commit = File.Exists(refPath) ? (await File.ReadAllTextAsync(refPath, ct).ConfigureAwait(false)).Trim()
                : LooksLikeCommit(revision) ? revision : null;
            if (commit is null)
                throw new InvalidOperationException(
                    $"'{repoId}' is not in the local cache ({CacheDir}) and the Hugging Face Hub could not be reached.");
            var snap = Path.Combine(repoDir, "snapshots", commit);
            var missing = fileList.Where(f => !File.Exists(Path.Combine(snap, f))).ToList();
            if (missing.Count > 0)
                throw new InvalidOperationException(
                    $"'{repoId}' is cached at {snap} but is missing {string.Join(", ", missing)}, and the Hub could not be reached.");
            return snap;
        }

        Directory.CreateDirectory(Path.Combine(repoDir, "refs"));
        if (!LooksLikeCommit(revision) || revision != commit)
            await File.WriteAllTextAsync(Path.Combine(repoDir, "refs", revision), commit, ct).ConfigureAwait(false);

        var snapshot = Path.Combine(repoDir, "snapshots", commit);
        foreach (var file in fileList)
        {
            var dest = Path.Combine(snapshot, file);
            if (File.Exists(dest)) continue;
            await DownloadFileAsync(repoId, file, commit, dest, progress, ct).ConfigureAwait(false);
        }
        return snapshot;
    }

    /// <summary>True when <paramref name="file"/> exists in the repo at <paramref name="revision"/>.</summary>
    public async Task<bool> ExistsAsync(string repoId, string file, string revision = "main", CancellationToken ct = default)
    {
        using var req = Request(HttpMethod.Head, repoId, file, revision);
        using var resp = await _noRedirect.SendAsync(req, ct).ConfigureAwait(false);
        return resp.IsSuccessStatusCode || (int)resp.StatusCode is >= 300 and < 400;
    }

    private async Task<string?> TryResolveCommitAsync(string repoId, string file, string revision, CancellationToken ct)
    {
        try
        {
            using var req = Request(HttpMethod.Head, repoId, file, revision);
            using var resp = await _noRedirect.SendAsync(req, ct).ConfigureAwait(false);
            if (resp.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
                throw new UnauthorizedAccessException(
                    $"the Hub refused access to '{repoId}' ({(int)resp.StatusCode}); set HF_TOKEN or pass a token for gated or private repos.");
            if (resp.StatusCode == HttpStatusCode.NotFound)
                throw new FileNotFoundException($"'{file}' was not found in '{repoId}' at revision '{revision}'.");
            return resp.Headers.TryGetValues("X-Repo-Commit", out var v) ? v.FirstOrDefault() : null;
        }
        catch (HttpRequestException)
        {
            return null;
        }
    }

    private async Task DownloadFileAsync(string repoId, string file, string commit, string dest,
        IProgress<(string, long, long?)>? progress, CancellationToken ct)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
        var tmp = dest + ".incomplete";
        using var req = Request(HttpMethod.Get, repoId, file, commit);
        using var resp = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        if (!resp.IsSuccessStatusCode)
            throw new HttpRequestException($"downloading '{file}' from '{repoId}' failed: {(int)resp.StatusCode} {resp.ReasonPhrase}");
        var total = resp.Content.Headers.ContentLength;
        await using (var src = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
        await using (var dst = File.Create(tmp))
        {
            var buffer = new byte[1 << 20];
            long done = 0;
            int n;
            while ((n = await src.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
            {
                await dst.WriteAsync(buffer.AsMemory(0, n), ct).ConfigureAwait(false);
                done += n;
                progress?.Report((file, done, total));
            }
        }
        File.Move(tmp, dest, overwrite: true);
    }

    private HttpRequestMessage Request(HttpMethod method, string repoId, string file, string revision)
    {
        var path = string.Join('/', file.Split('/').Select(Uri.EscapeDataString));
        var req = new HttpRequestMessage(method, $"{_endpoint}/{repoId}/resolve/{Uri.EscapeDataString(revision)}/{path}");
        if (!string.IsNullOrEmpty(_token)) req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _token);
        req.Headers.UserAgent.ParseAdd("NLaya/0.1");
        return req;
    }

    private static bool LooksLikeCommit(string s) => s.Length == 40 && s.All(Uri.IsHexDigit);

    public void Dispose()
    {
        _http.Dispose();
        _noRedirect.Dispose();
    }
}
