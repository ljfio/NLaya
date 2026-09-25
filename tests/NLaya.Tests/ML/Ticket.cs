namespace NLaya.Tests.ML;

/// <summary>An input row: the text Laya reads, plus columns that pass through.</summary>
public sealed class Ticket
{
    public string Body { get; set; } = "";
    public int Id { get; set; }
    public float Amount { get; set; }
}
