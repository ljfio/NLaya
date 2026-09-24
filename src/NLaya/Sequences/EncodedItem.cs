namespace NLaya.Sequences;

/// <summary>One question row: token ids plus the [MASK] marker position of each option.</summary>
public sealed record EncodedItem(int[] Ids, int[] Markers, QuestionType Type);
