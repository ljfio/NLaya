using System.Runtime.CompilerServices;

using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>
/// Screens each request with Laya before the inner client sees it. Port of Python's
/// <c>LayaGuardrail</c>: noul and score answers at or above their threshold are violations, and
/// <see cref="LayaGuardrailOptions.Action"/> decides whether to throw, answer with a refusal, or annotate.
/// Streaming requests are checked once, before the first update.
/// </summary>
public sealed class LayaGuardrailChatClient : DelegatingChatClient
{
    private readonly ILayaPredictor _predictor;
    private readonly LayaGuardrailOptions _options;
    private readonly Questions _questions;

    public LayaGuardrailChatClient(IChatClient innerClient, ILayaPredictor predictor, LayaGuardrailOptions? options = null)
        : base(innerClient)
    {
        _predictor = predictor ?? throw new ArgumentNullException(nameof(predictor));
        _options = options ?? new LayaGuardrailOptions();
        _questions = _options.Questions ?? Presets.Guard();
    }

    /// <summary>Run the guardrail questions over <paramref name="messages"/>; null when there is no text to screen.</summary>
    public async Task<GuardrailResult?> CheckAsync(IEnumerable<ChatMessage> messages, CancellationToken cancellationToken = default)
    {
        var text = _options.SelectText is { } select ? select(messages) : ChatText.LatestUserText(messages);
        if (string.IsNullOrWhiteSpace(text)) return null;
        var result = await _predictor.PredictAsync(text, _questions, _options.PredictOptions, cancellationToken).ConfigureAwait(false);
        return new GuardrailResult(Violations(result), result);
    }

    private List<GuardrailViolation> Violations(LayaResult result)
    {
        var found = new List<GuardrailViolation>();
        foreach (var (id, answer) in result.Answers)
        {
            var threshold = _options.Thresholds.TryGetValue(id, out var t) ? t : _options.Threshold;
            var value = answer switch
            {
                NoulAnswer n => n.Probability,
                ScoreAnswer s => s.Score,
                _ => (double?)null,
            };
            if (value >= threshold) found.Add(new GuardrailViolation(id, answer.Type, value.Value, threshold, answer.Confidence));
        }
        return found;
    }

    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var list = messages as IList<ChatMessage> ?? messages.ToList();
        var check = await CheckAsync(list, cancellationToken).ConfigureAwait(false);
        if (Blocked(check) is { } refusal)
            return new ChatResponse(new ChatMessage(ChatRole.Assistant, refusal))
            {
                FinishReason = ChatFinishReason.ContentFilter,
                AdditionalProperties = new() { [LayaChatProperties.Guardrail] = check },
            };

        var response = await base.GetResponseAsync(list, options, cancellationToken).ConfigureAwait(false);
        if (check is not null && _options.Action == GuardrailAction.Annotate)
            (response.AdditionalProperties ??= [])[LayaChatProperties.Guardrail] = check;
        return response;
    }

    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var list = messages as IList<ChatMessage> ?? messages.ToList();
        var check = await CheckAsync(list, cancellationToken).ConfigureAwait(false);
        if (Blocked(check) is { } refusal)
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, refusal)
            {
                FinishReason = ChatFinishReason.ContentFilter,
                AdditionalProperties = new() { [LayaChatProperties.Guardrail] = check },
            };
            yield break;
        }

        var first = check is not null && _options.Action == GuardrailAction.Annotate;
        await foreach (var update in base.GetStreamingResponseAsync(list, options, cancellationToken).ConfigureAwait(false))
        {
            if (first)
            {
                (update.AdditionalProperties ??= [])[LayaChatProperties.Guardrail] = check;
                first = false;
            }
            yield return update;
        }
    }

    /// <summary>The refusal to send instead of calling the inner client, or null to go ahead; throws for <see cref="GuardrailAction.Raise"/>.</summary>
    private string? Blocked(GuardrailResult? check)
    {
        if (check is null || check.Passed) return null;
        return _options.Action switch
        {
            GuardrailAction.Raise => throw new LayaGuardrailException(check),
            GuardrailAction.Filter => _options.RejectionMessage,
            _ => null,
        };
    }
}
