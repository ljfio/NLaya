using Microsoft.ML.Data;

namespace NLaya.Tests.ML;

/// <summary>A scored row read back with <c>CreateEnumerable</c>, as the README shows.</summary>
public sealed class ScoredTicket
{
    public int Id { get; set; }
    [ColumnName("team")] public string Team { get; set; } = "";
    [ColumnName("team_probs")] public float[] TeamProbabilities { get; set; } = [];
    [ColumnName("refund")] public bool Refund { get; set; }
}
