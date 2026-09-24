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

    public static EncodedItem Build(LayaTokenizer tok, int[] stateIds, Question q, int maxLen, int headMaxLen, bool truncateLeft)
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

        var ids = new List<int>(Math.Min(maxLen, headLen + stateIds.Length + 256)) { tok.ClsId };
        ids.AddRange(headIds.AsSpan(0, headLen));
        ids.Add(tok.SepId);
        var markers = new List<int>(optIds.Count);
        foreach (var o in optIds)
        {
            markers.Add(ids.Count);
            ids.AddRange(o);
        }
        ids.Add(tok.SepId);

        var room = Math.Max(0, maxLen - ids.Count - 1);
        var take = Math.Min(room, stateIds.Length);
        // Conversations keep their newest (last) turns; everything else keeps its beginning.
        var from = truncateLeft ? stateIds.Length - take : 0;
        ids.AddRange(stateIds.AsSpan(from, take));
        ids.Add(tok.SepId);

        var outIds = ids.Count > maxLen ? ids.GetRange(0, maxLen).ToArray() : ids.ToArray();
        return new EncodedItem(outIds, markers.Where(m => m < maxLen).ToArray(), q.Type);
    }

    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);
}
