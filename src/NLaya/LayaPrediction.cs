namespace NLaya;

/// <summary>One streamed input and its answers: <c>await foreach (var (ticket, result) in ...)</c>.</summary>
public sealed record LayaPrediction<T>(T Item, LayaResult Result);
