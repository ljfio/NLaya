namespace NLaya;

/// <summary>The three typed primitives the decision head answers.</summary>
public enum QuestionType
{
    /// <summary>Pick one label from a set.</summary>
    Choice = 0,
    /// <summary>An ordinal level, index 0 first; the answer is the expected level.</summary>
    Score = 1,
    /// <summary>A boolean statement; the answer is P(true).</summary>
    Noul = 2,
}
