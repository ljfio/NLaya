namespace NLaya.Backends;

/// <summary>
/// Raw model outputs: <c>Logits</c> is <c>[Rows, MaxMarkers]</c> (missing options already masked to
/// -1e4), <c>ActLogits</c> is <c>[Rows, ActOutputs]</c>. Softmax and temperature happen in the agent.
/// </summary>
public sealed record BackendOutput(float[] Logits, float[] ActLogits, int Rows, int MaxMarkers, int ActOutputs);
