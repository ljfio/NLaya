namespace NLaya.Parity;

/// <summary>
/// One <see cref="ParityFixture"/> for every model test class, run one class at a time: each checkpoint is
/// loaded once, which keeps the run inside a CI runner's memory.
/// </summary>
[CollectionDefinition(Name)]
public sealed class ParityCollection : ICollectionFixture<ParityFixture>
{
    public const string Name = "Parity";
}
