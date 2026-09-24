using System.Text.Json;

using Microsoft.Extensions.AI;

using NLaya.Extensions.AI;

namespace NLaya.Tests.ExtensionsAI;

public class LayaToolsTests
{
    [Fact]
    public async Task Triage_tool_runs_the_triage_preset_and_returns_the_answers_as_json()
    {
        var predictor = new FakePredictor((_, _) => new() { ["intent"] = FakePredictor.Choice("refund") });
        var tool = LayaTools.Triage(predictor);

        Assert.Equal("laya_triage", tool.Name);
        Assert.Contains("\"text\"", tool.JsonSchema.GetRawText());
        var result = await tool.InvokeAsync(new AIFunctionArguments { ["text"] = "refund me" }, TestContext.Current.CancellationToken);

        Assert.Equal("refund me", predictor.Calls[0].Text);
        Assert.Equal(Presets.Triage().Keys, predictor.Calls[0].Questions.Keys);
        var json = Assert.IsType<JsonElement>(result);
        Assert.Equal("refund", json.GetProperty("answers").GetProperty("intent").GetProperty("choice").GetString());
    }
}
