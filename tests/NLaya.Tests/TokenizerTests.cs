namespace NLaya.Tests;

public class TokenizerTests
{
    public static TheoryData<string, int> Cases()
    {
        var data = new TheoryData<string, int>();
        foreach (var name in new[] { "multilingual", "english" })
        {
            var n = TestFiles.Fixture($"tokenizer_{name}.json")["cases"]!.AsArray().Count;
            for (var i = 0; i < n; i++) data.Add(name, i);
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Matches_hf_tokenizers(string name, int i)
    {
        var fixture = TestFiles.Fixture($"tokenizer_{name}.json");
        var tok = TestFiles.Tokenizer(fixture["repo"]!.GetValue<string>());
        var c = fixture["cases"]![i]!;
        var expected = c["ids"]!.AsArray().Select(x => x!.GetValue<int>()).ToArray();
        Assert.Equal(expected, tok.Encode(c["text"]!.GetValue<string>()));
    }

    [Fact]
    public void Multilingual_special_ids()
    {
        var tok = TestFiles.Tokenizer(Laya.MultilingualModel);
        Assert.Equal((2, 1, 4, 0, "<mask>"), (tok.ClsId, tok.SepId, tok.MaskId, tok.PadId, tok.MaskToken));
    }

    [Fact]
    public void English_special_ids()
    {
        var tok = TestFiles.Tokenizer(Laya.DefaultModel);
        Assert.Equal((50281, 50282, 50284, 50283, "[MASK]"), (tok.ClsId, tok.SepId, tok.MaskId, tok.PadId, tok.MaskToken));
    }
}
