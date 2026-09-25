namespace NLaya.Sequences;

/// <summary>
/// The state-independent start of a question row, <c>[CLS] head [SEP] [MASK] opt0 ... [SEP]</c>,
/// tokenized once per call and shared by every state.
/// </summary>
internal sealed record EncodedQuestion(int[] Prefix, int[] Markers, QuestionType Type);
