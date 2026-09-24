using NLaya.Sequences;

namespace NLaya.Tests;

public class SequenceTests
{
    public static TheoryData<int> Cases() => new(Enumerable.Range(0, TestFiles.Fixture("sequences.json")["cases"]!.AsArray().Count));

    [Theory]
    [MemberData(nameof(Cases))]
    public void Matches_python_build_sequence(int i)
    {
        var c = TestFiles.Fixture("sequences.json")["cases"]![i]!;
        var tok = TestFiles.Tokenizer(Laya.MultilingualModel);
        var q = Question.FromJson(c["question"], c["qid"]!.GetValue<string>());
        var state = LayaState.FromJson(c["state"]?.DeepClone());
        var item = SequenceBuilder.Build(tok, SequenceBuilder.EncodeState(tok, state), q,
            c["max_len"]!.GetValue<int>(), c["head_max_len"]!.GetValue<int>(), state.IsConversation);
        Assert.Equal(c["ids"]!.AsArray().Select(x => x!.GetValue<int>()), item.Ids);
        Assert.Equal(c["markers"]!.AsArray().Select(x => x!.GetValue<int>()), item.Markers);
    }
}
