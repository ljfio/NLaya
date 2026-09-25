using System.Runtime.CompilerServices;

using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>
/// Sends each request to one of several chat clients, chosen by a Laya <c>choice</c> question over
/// the route descriptions. Port of Python's <c>LayaRouter</c>. The chosen label and the Laya result
/// are on the response's <c>AdditionalProperties</c> (<see cref="LayaChatProperties"/>). The route
/// clients belong to the caller and are not disposed.
/// </summary>
public sealed class LayaRouterChatClient : IChatClient
{
    /// <summary>The question id, as in Python.</summary>
    public const string QuestionId = "route";

    private readonly ILayaPredictor _predictor;
    private readonly LayaRouterChatClientOptions _options;
    private readonly Dictionary<string, IChatClient> _clients;
    private readonly Questions _questions;

    /// <summary>Route between <see cref="LayaRouterChatClientOptions.Routes"/> with <paramref name="predictor"/>.</summary>
    public LayaRouterChatClient(ILayaPredictor predictor, LayaRouterChatClientOptions options)
    {
        _predictor = predictor ?? throw new ArgumentNullException(nameof(predictor));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (options.Routes.Count == 0) throw new ArgumentException("add at least one route", nameof(options));
        if (options.Fallback is { } f && !options.Routes.ContainsKey(f))
            throw new ArgumentException($"fallback '{f}' is not one of the routes", nameof(options));
        _clients = options.Routes.ToDictionary(kv => kv.Key, kv => kv.Value.Client, StringComparer.Ordinal);
        _questions = new Questions
        {
            [QuestionId] = Question.Choice(options.Instructions,
                options.Routes.Select(kv => new KeyValuePair<string, string?>(kv.Key, kv.Value.Description))),
        };
    }

    /// <summary>Choose a route for <paramref name="messages"/> without calling it.</summary>
    public async Task<(string Route, LayaResult Result)> RouteAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var text = _options.SelectText is { } select ? select(messages) : ChatText.LatestUserText(messages);
        var result = await _predictor.PredictAsync(text ?? "", _questions, _options.PredictOptions, cancellationToken).ConfigureAwait(false);
        var answer = result.Choice(QuestionId);
        var confidence = _options.ConfidenceMeasure == RouteConfidence.Answer ? answer.AnswerConfidence : answer.Confidence;
        var route = _options.ConfidenceThreshold > 0 && confidence < _options.ConfidenceThreshold && _options.Fallback is { } fallback
            ? fallback
            : answer.Choice;
        return (route, result);
    }

    /// <summary>Choose a route, then answer with its client; the route and Laya result are on the response.</summary>
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var list = messages as IList<ChatMessage> ?? messages.ToList();
        var (route, result) = await RouteAsync(list, cancellationToken).ConfigureAwait(false);
        var response = await _clients[route].GetResponseAsync(list, options, cancellationToken).ConfigureAwait(false);
        Annotate(response.AdditionalProperties ??= [], route, result);
        return response;
    }

    /// <summary>Choose a route, then stream from its client; the route and Laya result are on the first update.</summary>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var list = messages as IList<ChatMessage> ?? messages.ToList();
        var (route, result) = await RouteAsync(list, cancellationToken).ConfigureAwait(false);
        var first = true;
        await foreach (var update in _clients[route].GetStreamingResponseAsync(list, options, cancellationToken).ConfigureAwait(false))
        {
            if (first)
            {
                Annotate(update.AdditionalProperties ??= [], route, result);
                first = false;
            }
            yield return update;
        }
    }

    private static void Annotate(AdditionalPropertiesDictionary properties, string route, LayaResult result)
    {
        properties[LayaChatProperties.Route] = route;
        properties[LayaChatProperties.Result] = result;
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        if (serviceType.IsInstanceOfType(this)) return this;
        return serviceType.IsInstanceOfType(_predictor) ? _predictor : null;
    }

    /// <summary>Nothing to release: the route clients belong to the caller.</summary>
    public void Dispose() { }
}
