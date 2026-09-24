namespace NLaya.Sequences;

/// <summary>A state's question rows, and the state's position in the caller's input.</summary>
internal sealed record EncodedState(int Index, List<EncodedItem> Rows);
