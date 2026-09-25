namespace NLaya;

/// <summary>A typed decision with its evidence: the <typeparamref name="T"/> plus Python's <c>DecisionResult</c> details.</summary>
public sealed class DecisionResult<T> : DecisionResult
{
    /// <summary>The decided value.</summary>
    public required T Value { get; init; }
}
