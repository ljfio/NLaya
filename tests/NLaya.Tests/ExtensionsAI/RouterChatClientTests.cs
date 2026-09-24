using Microsoft.Extensions.AI;

using NLaya.Extensions.AI;

namespace NLaya.Tests.ExtensionsAI;

public class RouterChatClientTests
{
    private readonly EchoChatClient _simple = new("simple");
    private readonly EchoChatClient _complex = new("complex");

    private LayaRouterChatClient Build(ChoiceAnswer answer, Action<LayaRouterChatClientOptions>? configure = null) =>
        Build(new FakePredictor((_, _) => new() { [LayaRouterChatClient.QuestionId] = answer }), configure);

    private LayaRouterChatClient Build(FakePredictor predictor, Action<LayaRouterChatClientOptions>? configure = null)
    {
        var options = new LayaRouterChatClientOptions
        {
            Routes =
            {
                ["simple"] = (_simple, "greetings, FAQs, short factual questions"),
                ["complex"] = (_complex, "multi-step reasoning, code, analysis"),
            },
        };
        configure?.Invoke(options);
        return new LayaRouterChatClient(predictor, options);
    }

    [Fact]
    public async Task Sends_the_request_to_the_chosen_route_and_annotates_the_response()
    {
        var response = await Build(FakePredictor.Choice("complex")).GetResponseAsync("prove it", cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("complex: prove it", response.Text);
        Assert.Equal((0, 1), (_simple.Calls, _complex.Calls));
        Assert.Equal("complex", response.AdditionalProperties![LayaChatProperties.Route]);
        Assert.IsType<LayaResult>(response.AdditionalProperties[LayaChatProperties.Result]);
    }

    [Fact]
    public async Task Asks_one_choice_question_with_the_routes_in_order()
    {
        var predictor = new FakePredictor((_, _) => new() { ["route"] = FakePredictor.Choice("simple") });
        await Build(predictor, o => o.Instructions = "Pick one").GetResponseAsync("hi", cancellationToken: TestContext.Current.CancellationToken);
        var q = Assert.Single(predictor.Calls).Questions;
        Assert.Equal("""{"route":{"type":"choice","instructions":"Pick one","criteria":{"simple":"greetings, FAQs, short factual questions","complex":"multi-step reasoning, code, analysis"}}}""",
            q.ToJson().ToJsonString());
    }

    [Theory]
    [InlineData(RouteConfidence.Entropy, 0.4, 0.9, "complex")] // entropy 0.4 < 0.6: fall back
    [InlineData(RouteConfidence.Entropy, 0.7, 0.3, "simple")]
    [InlineData(RouteConfidence.Answer, 0.9, 0.5, "complex")]  // answer 0.5 < 0.6: fall back
    [InlineData(RouteConfidence.Answer, 0.1, 0.7, "simple")]
    public async Task Falls_back_below_the_confidence_threshold(RouteConfidence measure, double entropy, double answer, string expected)
    {
        var client = Build(FakePredictor.Choice("simple", entropy, answer), o =>
        {
            o.Fallback = "complex";
            o.ConfidenceThreshold = 0.6;
            o.ConfidenceMeasure = measure;
        });
        var (route, _) = await client.RouteAsync([new(ChatRole.User, "hi")], TestContext.Current.CancellationToken);
        Assert.Equal(expected, route);
    }

    [Fact]
    public async Task Keeps_the_choice_without_a_fallback()
    {
        var client = Build(FakePredictor.Choice("simple", 0.01, 0.01), o => o.ConfidenceThreshold = 0.6);
        var (route, _) = await client.RouteAsync([new(ChatRole.User, "hi")], TestContext.Current.CancellationToken);
        Assert.Equal("simple", route);
    }

    [Fact]
    public void Rejects_an_unknown_fallback_and_no_routes()
    {
        Assert.Throws<ArgumentException>(() => Build(FakePredictor.Choice("simple"), o => o.Fallback = "nope"));
        Assert.Throws<ArgumentException>(() => new LayaRouterChatClient(new FakePredictor((_, _) => new()), new LayaRouterChatClientOptions()));
    }

    [Fact]
    public async Task Streams_from_the_chosen_route()
    {
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in Build(FakePredictor.Choice("simple")).GetStreamingResponseAsync("hi", cancellationToken: TestContext.Current.CancellationToken))
            updates.Add(u);
        Assert.Equal("simple: hi", string.Concat(updates.Select(u => u.Text)));
        Assert.Equal("simple", updates[0].AdditionalProperties![LayaChatProperties.Route]);
    }
}
