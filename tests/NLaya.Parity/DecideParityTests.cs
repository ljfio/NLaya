using System.Text.Json.Nodes;

namespace NLaya.Parity;

/// <summary><c>Decide</c> on real checkpoints against Python's <c>Agent.decide(..., return_details=True)</c> (<c>decide.json</c>).</summary>
[Collection(ParityCollection.Name)]
[TestCaseOrderer(typeof(ByModelOrderer))]
public class DecideParityTests(ParityFixture fx)
{
    private static readonly JsonNode Fixture = ParityFixture.Fixture("decide.json");

    public static TheoryData<string, string, int> Cases()
    {
        var data = new TheoryData<string, string, int>();
        foreach (var model in ParityFixture.Models)
        {
            var n = Fixture["cases"]!.AsArray().Count(c => c!["model"]!.GetValue<string>() == model);
            foreach (var backend in new[] { "torchsharp", "onnx" })
                for (var i = 0; i < n; i++) data.Add(backend, model, i);
        }
        return data;
    }

    private static JsonNode Case(string model, int i) =>
        Fixture["cases"]!.AsArray().Where(c => c!["model"]!.GetValue<string>() == model).ElementAt(i)!;

    [Theory]
    [MemberData(nameof(Cases))]
    public void Json_schema_decides_like_python(string backend, string model, int i)
    {
        var c = Case(model, i);
        var agent = fx.Agent(backend, model);
        var details = agent.DecideWithDetails(LayaState.FromJson(c["state"]?.DeepClone()), Fixture["schema"]!);
        ModelParityTests.AssertResult(c["details"]!, details.ToJson());
        Assert.True(JsonNode.DeepEquals(c["details"]!["values"], details.Values), $"values: {details.Values.ToJsonString()}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Typed_decision_decides_like_python(string backend, string model, int i)
    {
        var c = Case(model, i);
        var agent = fx.Agent(backend, model);
        var ticket = agent.Decide(LayaState.FromJson(c["state"]?.DeepClone()), DecideContext.Default.DecideTicket);
        var want = c["details"]!["values"]!;
        Assert.Equal(want["department"]!.GetValue<string>(), ticket.Department.ToString());
        Assert.Equal(want["urgency"]!.GetValue<int>(), ticket.Urgency);
        Assert.Equal(want["refund_requested"]!.GetValue<bool>(), ticket.RefundRequested);
        Assert.Equal(want["needs_human"]!.GetValue<bool>(), ticket.NeedsHuman);
    }
}
