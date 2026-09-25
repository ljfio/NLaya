using NLaya.Tokenization;

namespace NLaya.Sequences;

/// <summary>
/// Port of <c>laya.common.build_sequence</c>:
/// <c>[CLS] "{type} question: {ins}" [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]</c>.
/// </summary>
public static class SequenceBuilder
{
    /// <summary>Longest option text, in tokens (the <c>max_length=48</c> in Python).</summary>
    public const int MaxOptionTokens = 48;

    /// <summary>Token ids of a state, tokenized once and shared by every question.</summary>
    public static int[] EncodeState(LayaTokenizer tok, LayaState state) =>
        tok.Encode(state.Serialize().Replace(tok.MaskToken, " ", StringComparison.Ordinal));

    /// <summary>One question row for a tokenized state, as <c>build_sequence</c> builds it.</summary>
    public static EncodedItem Build(LayaTokenizer tok, int[] stateIds, Question q, int maxLen, int headMaxLen, bool truncateLeft) =>
        Assemble(tok, EncodeQuestion(tok, q, headMaxLen), stateIds, maxLen, truncateLeft);

    /// <summary>The question's half of the row; it doesn't depend on the state, so batches encode it once.</summary>
    internal static EncodedQuestion EncodeQuestion(LayaTokenizer tok, Question q, int headMaxLen)
    {
        var mask = tok.MaskToken;
        var opts = q.RenderOptions();
        var ins = q.InstructionText.Replace(mask, " ", StringComparison.Ordinal);
        var headIds = tok.Encode($"{q.TypeName} question: {ins}");

        var optIds = new List<int[]>(opts.Count);
        foreach (var opt in opts)
        {
            var t = tok.Encode(" " + opt.Replace(mask, " ", StringComparison.Ordinal));
            var len = Math.Min(t.Length, MaxOptionTokens);
            var o = new int[len + 1];
            o[0] = tok.MaskId;
            Array.Copy(t, 0, o, 1, len);
            optIds.Add(o);
        }

        var optBudget = headMaxLen - optIds.Sum(o => o.Length);
        if (optBudget < 16)
        {
            var per = Math.Max(4, FloorDiv(headMaxLen - 16, Math.Max(1, optIds.Count)));
            for (var i = 0; i < optIds.Count; i++)
                if (optIds[i].Length > per) optIds[i] = optIds[i][..per];
            optBudget = headMaxLen - optIds.Sum(o => o.Length);
        }
        var headLen = Math.Min(headIds.Length, Math.Max(8, optBudget));

        var prefix = new List<int>(headLen + optIds.Sum(o => o.Length) + 3) { tok.ClsId };
        prefix.AddRange(headIds.AsSpan(0, headLen));
        prefix.Add(tok.SepId);
        var markers = new int[optIds.Count];
        for (var i = 0; i < optIds.Count; i++)
        {
            markers[i] = prefix.Count;
            prefix.AddRange(optIds[i]);
        }
        prefix.Add(tok.SepId);
        return new EncodedQuestion(prefix.ToArray(), markers, q.Type);
    }

    /// <summary><c>prefix state [SEP]</c>, cut to <paramref name="maxLen"/>; markers past the cut are dropped.</summary>
    internal static EncodedItem Assemble(LayaTokenizer tok, EncodedQuestion q, int[] stateIds, int maxLen, bool truncateLeft)
    {
        var prefix = q.Prefix;
        var take = Math.Min(Math.Max(0, maxLen - prefix.Length - 1), stateIds.Length);
        // Conversations keep their newest (last) turns; everything else keeps its beginning.
        var from = truncateLeft ? stateIds.Length - take : 0;

        var ids = new int[Math.Min(maxLen, prefix.Length + take + 1)];
        prefix.AsSpan(0, Math.Min(prefix.Length, ids.Length)).CopyTo(ids);
        if (ids.Length > prefix.Length)
        {
            stateIds.AsSpan(from, take).CopyTo(ids.AsSpan(prefix.Length));
            ids[^1] = tok.SepId;
        }
        var markers = q.Markers.Length > 0 && q.Markers[^1] >= maxLen ? q.Markers.Where(m => m < maxLen).ToArray() : q.Markers;
        return new EncodedItem(ids, markers, q.Type);
    }

    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);
}
