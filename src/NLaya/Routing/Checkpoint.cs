namespace NLaya.Routing;

/// <summary>
/// The three Laya checkpoints a <see cref="Router"/> chooses between. <see cref="Checkpoints.Name"/> gives
/// Python's name ("english", "multilingual", "typed-decisions"); <see cref="Checkpoints.Parse"/> reads a
/// name or alias, for configuration.
/// </summary>
public enum Checkpoint
{
    /// <summary>ModernBERT-large, English only.</summary>
    English,

    /// <summary>mmBERT-base, 100+ languages and scripts.</summary>
    Multilingual,

    /// <summary>The typed-decisions workflows (customer service, invoices, security incidents, agent traces).</summary>
    TypedDecisions,
}
