namespace NLaya;

/// <summary>Collects a choice question's options for <see cref="Questions.Choice(string, string, Action{ChoiceBuilder})"/>.</summary>
public sealed class ChoiceBuilder
{
    private readonly List<KeyValuePair<string, string?>> _options = [];

    internal ChoiceBuilder() { }

    /// <summary>Add an option; <paramref name="description"/> is what the model reads after the label.</summary>
    public ChoiceBuilder Option(string label, string? description = null)
    {
        ArgumentNullException.ThrowIfNull(label);
        _options.Add(new(label, description));
        return this;
    }

    internal Question Build(string instructions) => Question.Choice(instructions, _options);
}
