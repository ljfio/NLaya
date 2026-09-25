namespace NLaya;

/// <summary>Optional descriptions and labels for <see cref="Questions.Noul(string, string, Action{NoulBuilder}?)"/>.</summary>
public sealed class NoulBuilder
{
    private string? _whenTrue;
    private string? _whenFalse;
    private NoulLabels? _labels;

    internal NoulBuilder() { }

    /// <summary>What "true" means for this statement (defaults to "yes, the statement holds").</summary>
    public NoulBuilder WhenTrue(string description)
    {
        _whenTrue = description;
        return this;
    }

    /// <summary>What "false" means for this statement (defaults to "no, the statement does not hold").</summary>
    public NoulBuilder WhenFalse(string description)
    {
        _whenFalse = description;
        return this;
    }

    /// <summary>Display labels for the two options; the answer is still P(true).</summary>
    public NoulBuilder Labels(string falseLabel, string trueLabel)
    {
        _labels = new NoulLabels(falseLabel, trueLabel);
        return this;
    }

    internal Question Build(string instructions) => Question.Noul(instructions, _whenTrue, _whenFalse, _labels);
}
