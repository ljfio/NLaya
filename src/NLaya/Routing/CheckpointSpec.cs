namespace NLaya.Routing;

/// <summary>A checkpoint location: a Hub repo (or local path) and an optional subfolder.</summary>
public sealed record CheckpointSpec(string Repo, string? Subfolder = null)
{
    /// <summary><c>repo</c> or <c>repo/subfolder</c>.</summary>
    public override string ToString() => Subfolder is null ? Repo : $"{Repo}/{Subfolder}";
}
