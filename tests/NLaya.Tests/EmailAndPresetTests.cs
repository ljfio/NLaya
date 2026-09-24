using System.Text.Json.Nodes;
using NLaya.Email;

namespace NLaya.Tests;

public class EmailAndPresetTests
{
    public static TheoryData<int> Cases() => new(Enumerable.Range(0, TestFiles.Fixture("email.json")["cases"]!.AsArray().Count));

    [Theory]
    [MemberData(nameof(Cases))]
    public void Cleans_like_python(int i)
    {
        var c = TestFiles.Fixture("email.json")["cases"]![i]!;
        var body = c["body"]!.GetValue<string>();
        Assert.Equal(c["clean"]!.GetValue<string>(), EmailCleaner.CleanBody(body));
        JsonAssert.Equivalent(c["state"], EmailCleaner.State("Subject line", body, sender: "a@b.com"));
    }

    [Fact]
    public void Raw_state_keeps_body_and_extras()
    {
        var expected = TestFiles.Fixture("email.json")["raw_state"];
        JsonAssert.Equivalent(expected, EmailCleaner.State("S", "  body  ", clean: false,
            extra: [new("priority", JsonValue.Create("high"))]));
    }

    [Fact]
    public void Presets_round_trip_the_python_definitions()
    {
        var src = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "../../../../../src/NLaya/Presets/presets.json")))!;
        JsonAssert.Equivalent(src["triage_questions"], Presets.Triage().ToJson());
        JsonAssert.Equivalent(src["email_questions"], Presets.Email().ToJson());
        JsonAssert.Equivalent(src["guard_questions"], Presets.Guard().ToJson());
        JsonAssert.Equivalent(src["moderation_questions"], Presets.Moderation().ToJson());
        JsonAssert.Equivalent(src["router_questions"], Presets.Router().ToJson());
        var custom = Presets.Email([new("sales", "buying"), new("other", null)]);
        Assert.Equal(["sales", "other"], custom["category"].Options.Select(o => o.Key));
    }
}
