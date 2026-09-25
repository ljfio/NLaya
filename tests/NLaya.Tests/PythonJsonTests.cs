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

    [Fact]
    public void Serializes_other_clr_values_without_reflection()
    {
        var node = new JsonObject { ["l"] = 9_000_000_000L, ["c"] = 'é', ["g"] = new Guid("0f8fad5b-d9cb-469f-a165-70867728950e"), ["u"] = "日本\n" };
        Assert.Equal("{\"l\": 9000000000, \"c\": \"é\", \"g\": \"0f8fad5b-d9cb-469f-a165-70867728950e\", \"u\": \"日本\\n\"}", PythonJson.Serialize(node));
    }

    [Fact]
    public void Source_generated_state_matches_reflection()
    {
        var ticket = new Ticket("héllo", 7, ["a", "b"]);
        var typed = LayaState.From(ticket, TestJsonContext.Default.Ticket);
        Assert.Equal(LayaState.From(ticket).Serialize(), typed.Serialize());
        Assert.Equal("{\"Body\": \"héllo\", \"Id\": 7, \"Tags\": [\"a\", \"b\"]}", typed.Serialize());
        Assert.Equal("hi", LayaState.From("hi", TestJsonContext.Default.String).Text);
    }

    [Fact]
    public void Conversations_from_states_and_typed_turns_match_reflection()
    {
        var turns = new[] { new Ticket("a", 1, []), new Ticket("b", 2, ["x"]) };
        var reflected = LayaState.Conversation(turns.Cast<object?>()).Serialize();
        Assert.Equal(reflected, LayaState.Conversation(turns, TestJsonContext.Default.Ticket).Serialize());
        Assert.Equal(reflected, LayaState.Conversation(turns.Select(t => LayaState.From(t, TestJsonContext.Default.Ticket))).Serialize());
        Assert.Equal("[\"hi\", {\"role\": \"user\"}]",
            LayaState.Conversation([(LayaState)"hi", LayaState.ParseJson("{\"role\": \"user\"}")]).Serialize());
    }
}
