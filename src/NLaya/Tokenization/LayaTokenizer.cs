using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Microsoft.ML.Tokenizers;

namespace NLaya.Tokenization;

/// <summary>
/// A checkpoint's Hugging Face <c>tokenizer.json</c>, run by <see cref="BpeTokenizer"/> from
/// Microsoft.ML.Tokenizers. This class only adds what that library has no option for:
/// added tokens matched longest-first, and for Metaspace (Gemma-style) vocabularies the per-segment
/// <c>▁</c> prefix and byte fallback. <see cref="Encode"/> never adds special tokens.
/// </summary>
public sealed class LayaTokenizer
{
    private const char Metaspace = '▁';
    private const string Gpt2Split = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    private readonly BpeTokenizer _bpe;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<string, int> _added;
    private readonly Regex? _addedRe;
    private readonly bool _metaspace;

    public int ClsId { get; }
    public int SepId { get; }
    public int MaskId { get; }
    public int PadId { get; }
    public string MaskToken { get; }

    private LayaTokenizer(JsonElement root, JsonElement? config)
    {
        var model = root.GetProperty("model");
        _vocab = model.GetProperty("vocab").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);
        var merges = model.GetProperty("merges").EnumerateArray()
            .Select(m => m.ValueKind == JsonValueKind.String ? m.GetString()! : m[0].GetString() + " " + m[1].GetString());
        var added = root.GetProperty("added_tokens").EnumerateArray()
            .Select(a => (Content: a.GetProperty("content").GetString()!, Id: a.GetProperty("id").GetInt32(),
                LStrip: a.TryGetProperty("lstrip", out var l) && l.GetBoolean()))
            .Where(a => a.Content.Length > 0).ToList();
        _added = added.ToDictionary(a => a.Content, a => a.Id, StringComparer.Ordinal);
        foreach (var (content, id, _) in added) _vocab.TryAdd(content, id);
        if (added.Count > 0)
            _addedRe = new Regex(string.Join("|", added.OrderByDescending(a => a.Content.Length)
                .Select(a => (a.LStrip ? @"\s*" : "") + "(" + Regex.Escape(a.Content) + ")")), RegexOptions.Compiled);

        _metaspace = root.GetProperty("pre_tokenizer").GetRawText().Contains("\"Metaspace\"", StringComparison.Ordinal);
        var unk = model.TryGetProperty("unk_token", out var u) && u.ValueKind == JsonValueKind.String ? u.GetString() : null;
        _bpe = BpeTokenizer.Create(new BpeOptions(_vocab)
        {
            Merges = merges,
            UnknownToken = unk,
            FuseUnknownTokens = model.TryGetProperty("fuse_unk", out var f) && f.GetBoolean(),
            ByteLevel = !_metaspace,
            PreTokenizer = new RegexPreTokenizer(new Regex(_metaspace ? "▁[^▁]*|[^▁]+" : Gpt2Split, RegexOptions.Compiled), null),
        });

        (ClsId, _) = Special(config, "cls_token", "[CLS]", "<bos>", "<s>");
        (SepId, _) = Special(config, "sep_token", "[SEP]", "<eos>", "</s>");
        (PadId, _) = Special(config, "pad_token", "[PAD]", "<pad>");
        (MaskId, MaskToken) = Special(config, "mask_token", "[MASK]", "<mask>");
    }

    /// <summary>Load <c>tokenizer.json</c> (and optionally <c>tokenizer_config.json</c> for the special tokens).</summary>
    public static LayaTokenizer FromFile(string tokenizerJson, string? tokenizerConfigJson = null)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        using var cfg = tokenizerConfigJson is not null && File.Exists(tokenizerConfigJson)
            ? JsonDocument.Parse(File.ReadAllBytes(tokenizerConfigJson)) : null;
        return new LayaTokenizer(doc.RootElement, cfg?.RootElement);
    }

    /// <summary>Token ids for <paramref name="text"/>, without special tokens.</summary>
    public int[] Encode(string text)
    {
        var ids = new List<int>();
        var pos = 0;
        if (_addedRe is not null)
        {
            foreach (Match m in _addedRe.Matches(text))
            {
                EncodeSegment(text[pos..m.Index], ids);
                ids.Add(_added[m.Groups.Cast<Group>().Skip(1).First(g => g.Success).Value]);
                pos = m.Index + m.Length;
            }
        }
        EncodeSegment(text[pos..], ids);
        return ids.ToArray();
    }

    private void EncodeSegment(string s, List<int> ids)
    {
        if (s.Length == 0) return;
        if (!_metaspace)
        {
            ids.AddRange(_bpe.EncodeToIds(s.Normalize(NormalizationForm.FormC)));
            return;
        }
        s = s.Replace(' ', Metaspace);
        if (s[0] != Metaspace) s = Metaspace + s;
        // Byte fallback: a character missing from the vocab becomes its UTF-8 byte tokens.
        var run = new StringBuilder();
        foreach (var r in s.EnumerateRunes())
        {
            var ch = r.ToString();
            if (_vocab.ContainsKey(ch)) { run.Append(ch); continue; }
            if (run.Length > 0) { ids.AddRange(_bpe.EncodeToIds(run.ToString())); run.Clear(); }
            ids.AddRange(Encoding.UTF8.GetBytes(ch).Select(b => _vocab.TryGetValue($"<0x{b:X2}>", out var id) ? id : _vocab["<unk>"]));
        }
        if (run.Length > 0) ids.AddRange(_bpe.EncodeToIds(run.ToString()));
    }

    private (int Id, string Token) Special(JsonElement? config, string key, params string[] aliases)
    {
        if (config is { } c && c.TryGetProperty(key, out var v))
        {
            var content = v.ValueKind == JsonValueKind.String ? v.GetString()
                : v.ValueKind == JsonValueKind.Object && v.TryGetProperty("content", out var cc) ? cc.GetString() : null;
            if (content is not null && _vocab.TryGetValue(content, out var id)) return (id, content);
        }
        foreach (var a in aliases)
            if (_vocab.TryGetValue(a, out var id)) return (id, a);
        throw new InvalidDataException($"tokenizer has no {key} (tried {string.Join(", ", aliases)})");
    }
}
