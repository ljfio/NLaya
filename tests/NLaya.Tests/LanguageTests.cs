using NLaya.Lang;

namespace NLaya.Tests;

public class LanguageTests
{
    public static TheoryData<int> Cases() => new(Enumerable.Range(0, TestFiles.Fixture("lang.json")["cases"]!.AsArray().Count));

    [Theory]
    [MemberData(nameof(Cases))]
    public void Analyse_matches_python(int i)
    {
        var c = TestFiles.Fixture("lang.json")["cases"]![i]!;
        var det = LanguageDetector.Analyse(LayaState.FromJson(c["state"]?.DeepClone()));
        JsonAssert.Equivalent(c["analyse"], det.ToJson(), 1e-9);
    }
}
