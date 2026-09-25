using System.Diagnostics.CodeAnalysis;

namespace NLaya.Routing;

/// <summary>Names and aliases for <see cref="Checkpoint"/>, as Python's <c>laya.Router</c> spells them.</summary>
public static class Checkpoints
{
    private static readonly Dictionary<string, Checkpoint> ByName = new(StringComparer.Ordinal)
    {
        ["english"] = Checkpoint.English,
        ["multilingual"] = Checkpoint.Multilingual,
        ["typed-decisions"] = Checkpoint.TypedDecisions,
        ["en"] = Checkpoint.English,
        ["laya"] = Checkpoint.English,
        ["default"] = Checkpoint.English,
        ["multi"] = Checkpoint.Multilingual,
        ["ml"] = Checkpoint.Multilingual,
        ["laya-multilingual"] = Checkpoint.Multilingual,
        ["typed"] = Checkpoint.TypedDecisions,
        ["typed_decisions"] = Checkpoint.TypedDecisions,
        ["laya-typed-decisions"] = Checkpoint.TypedDecisions,
        ["decisions"] = Checkpoint.TypedDecisions,
    };

    /// <summary>Python's name: "english", "multilingual" or "typed-decisions".</summary>
    public static string Name(this Checkpoint checkpoint) => checkpoint switch
    {
        Checkpoint.English => "english",
        Checkpoint.Multilingual => "multilingual",
        Checkpoint.TypedDecisions => "typed-decisions",
        _ => throw new ArgumentOutOfRangeException(nameof(checkpoint), checkpoint, null),
    };

    /// <summary>A checkpoint from its name, an alias ("en", "ml", "typed_decisions") or the enum member name; case-insensitive.</summary>
    public static bool TryParse(string? name, [NotNullWhen(true)] out Checkpoint? checkpoint)
    {
        checkpoint = null;
        if (name is null) return false;
        var key = name.Trim().ToLowerInvariant();
        if (ByName.TryGetValue(key, out var c) || Enum.TryParse(key, ignoreCase: true, out c) && Enum.IsDefined(c))
            checkpoint = c;
        return checkpoint is not null;
    }

    /// <inheritdoc cref="TryParse"/>
    /// <exception cref="ArgumentException">Not a checkpoint name or alias; the message lists them, as Python's does.</exception>
    public static Checkpoint Parse(string name) => TryParse(name, out var c) ? c.Value : throw new ArgumentException(
        $"unknown model {PyStr.Repr(name)}; choose one of ['english', 'multilingual', 'typed-decisions'] " +
        $"(or an alias: [{string.Join(", ", ByName.Keys.Skip(3).Order(StringComparer.Ordinal).Select(PyStr.Repr))}])", nameof(name));
}
