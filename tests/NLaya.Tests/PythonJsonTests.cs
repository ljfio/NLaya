using System.Text.Json.Nodes;

namespace NLaya.Tests;

public class PythonJsonTests
{
    public static TheoryData<int> Cases() => new(Enumerable.Range(0, TestFiles.Fixture("json_states.json")["cases"]!.AsArray().Count));

    [Theory]
    [MemberData(nameof(Cases))]
    public void Matches_python_json_dumps(int i)
    {
        var c = TestFiles.Fixture("json_states.json")["cases"]![i]!;
        var state = LayaState.FromJson(c["state"]?.DeepClone());
        Assert.Equal(c["json"]!.GetValue<string>(), state.Serialize());
    }

    [Theory]
    [InlineData(3.14, "3.14")]
    [InlineData(2.0, "2.0")]
    [InlineData(1e20, "1e+20")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(1e-7, "1e-07")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(0.00001, "1e-05")]
    [InlineData(-0.5, "-0.5")]
    [InlineData(123456.789, "123456.789")]
    [InlineData(1.5e300, "1.5e+300")]
    [InlineData(0.1 + 0.2, "0.30000000000000004")]
    [InlineData(0.0, "0.0")]
    public void Formats_floats_like_python_repr(double value, string expected) =>
        Assert.Equal(expected, PythonJson.FormatFloat(value));

    [Fact]
    public void Serializes_clr_values_like_python()
    {
        var node = new JsonObject { ["i"] = 3, ["d"] = 3.0, ["f"] = 0.5f, ["s"] = "<a & 'b'>", ["b"] = false, ["n"] = null };
        Assert.Equal("{\"i\": 3, \"d\": 3.0, \"f\": 0.5, \"s\": \"<a & 'b'>\", \"b\": false, \"n\": null}", PythonJson.Serialize(node));
    }

    [Fact]
    public void Anonymous_objects_keep_property_order()
    {
        var s = LayaState.From(new { body = "héllo", id = 7, tags = new[] { "a", "b" } });
        Assert.Equal("{\"body\": \"héllo\", \"id\": 7, \"tags\": [\"a\", \"b\"]}", s.Serialize());
        Assert.False(s.IsConversation);
        Assert.True(LayaState.Conversation(["hi", new { role = "user" }]).IsConversation);
    }
}
