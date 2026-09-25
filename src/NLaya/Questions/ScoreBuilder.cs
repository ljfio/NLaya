namespace NLaya;

/// <summary>Collects a score question's levels for <see cref="Questions.Score(string, string, Action{ScoreBuilder})"/>.</summary>
public sealed class ScoreBuilder
{
    private readonly List<string> _levels = [];

    internal ScoreBuilder() { }

    /// <summary>Add the next level, lowest first: the first call describes level 0.</summary>
    public ScoreBuilder Level(string description)
    {
        ArgumentNullException.ThrowIfNull(description);
        _levels.Add(description);
        return this;
    }

    internal Question Build(string instructions) => Question.Score(instructions, _levels);
}
