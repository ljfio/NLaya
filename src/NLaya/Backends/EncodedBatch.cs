using NLaya.Sequences;

namespace NLaya.Backends;

/// <summary>
/// A padded batch of question rows, row-major. Mirrors <c>laya.common.collate_items</c>:
/// <c>input_ids</c>/<c>attention_mask</c> are <c>[Rows, SeqLen]</c>, <c>marker_pos</c>/<c>marker_mask</c>
/// are <c>[Rows, MaxMarkers]</c>, and <c>qtype</c> is <c>[Rows]</c>.
/// </summary>
public sealed class EncodedBatch
{
    /// <summary>Question rows in the batch (states × questions).</summary>
    public int Rows { get; }
    /// <summary>Padded sequence length: the longest row.</summary>
    public int SeqLen { get; }
    /// <summary>The most options any row has.</summary>
    public int MaxMarkers { get; }
    /// <summary>Token ids, <c>[Rows, SeqLen]</c>, padded with the tokenizer's pad id.</summary>
    public long[] InputIds { get; }
    /// <summary>1 for real tokens, 0 for padding, <c>[Rows, SeqLen]</c>.</summary>
    public long[] AttentionMask { get; }
    /// <summary>Position of each option's [MASK] marker, <c>[Rows, MaxMarkers]</c>.</summary>
    public long[] MarkerPos { get; }
    /// <summary>True where a row has that option, <c>[Rows, MaxMarkers]</c>.</summary>
    public bool[] MarkerMask { get; }
    /// <summary>Each row's question type (0 choice, 1 score, 2 noul), <c>[Rows]</c>.</summary>
    public long[] QType { get; }
    /// <summary>Real (unpadded) length of each row.</summary>
    public int[] Lengths { get; }
    /// <summary>Real number of options of each row.</summary>
    public int[] OptionCounts { get; }

    private EncodedBatch(int rows, int seqLen, int maxMarkers)
    {
        Rows = rows;
        SeqLen = seqLen;
        MaxMarkers = maxMarkers;
        InputIds = new long[rows * seqLen];
        AttentionMask = new long[rows * seqLen];
        MarkerPos = new long[rows * maxMarkers];
        MarkerMask = new bool[rows * maxMarkers];
        QType = new long[rows];
        Lengths = new int[rows];
        OptionCounts = new int[rows];
    }

    /// <summary>Pad <paramref name="items"/> into one batch, as <c>laya.common.collate_items</c> does.</summary>
    public static EncodedBatch Collate(IReadOnlyList<EncodedItem> items, int padId)
    {
        if (items.Count == 0) throw new ArgumentException("cannot collate an empty batch", nameof(items));
        var seqLen = items.Max(i => i.Ids.Length);
        var k = items.Max(i => i.Markers.Length);
        var b = new EncodedBatch(items.Count, seqLen, k);
        Array.Fill(b.InputIds, padId);
        for (var r = 0; r < items.Count; r++)
        {
            var it = items[r];
            for (var j = 0; j < it.Ids.Length; j++)
            {
                b.InputIds[r * seqLen + j] = it.Ids[j];
                b.AttentionMask[r * seqLen + j] = 1;
            }
            for (var j = 0; j < it.Markers.Length; j++)
            {
                b.MarkerPos[r * k + j] = it.Markers[j];
                b.MarkerMask[r * k + j] = true;
            }
            b.QType[r] = (int)it.Type;
            b.Lengths[r] = it.Ids.Length;
            b.OptionCounts[r] = it.Markers.Length;
        }
        return b;
    }

    /// <summary>Build directly from tensors (tests, fixtures).</summary>
    public static EncodedBatch FromArrays(long[][] inputIds, long[][] attentionMask, long[][] markerPos, bool[][] markerMask, long[] qtype)
    {
        var rows = inputIds.Length;
        var b = new EncodedBatch(rows, inputIds[0].Length, markerPos[0].Length);
        for (var r = 0; r < rows; r++)
        {
            inputIds[r].CopyTo(b.InputIds, r * b.SeqLen);
            attentionMask[r].CopyTo(b.AttentionMask, r * b.SeqLen);
            markerPos[r].CopyTo(b.MarkerPos, r * b.MaxMarkers);
            markerMask[r].CopyTo(b.MarkerMask, r * b.MaxMarkers);
            b.QType[r] = qtype[r];
            b.Lengths[r] = (int)attentionMask[r].Sum();
            b.OptionCounts[r] = markerMask[r].Count(m => m);
        }
        return b;
    }
}
