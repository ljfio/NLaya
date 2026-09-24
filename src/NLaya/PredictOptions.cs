namespace NLaya;

/// <summary>Per-call settings for <see cref="LayaAgent.Predict(LayaState, Questions, PredictOptions?)"/>.</summary>
public class PredictOptions
{
    /// <summary>Language code; selects a <see cref="LayaOptions.LangTemperatures"/> override.</summary>
    public string? Lang { get; init; }

    /// <summary>Token budget per question row. <c>laya-multilingual</c> reads up to 8192.</summary>
    public int? MaxLen { get; init; }

    /// <summary>Token budget for the question and its options.</summary>
    public int? HeadMaxLen { get; init; }

    /// <summary>Hooks for this call only, run after installed hooks.</summary>
    public IEnumerable<ILayaHook>? Hooks { get; init; }

    public bool? HooksRaise { get; init; }
}
