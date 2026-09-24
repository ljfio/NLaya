using System.Runtime.CompilerServices;

using Microsoft.Extensions.AI;

/// <summary>Stands in for a real LLM client (OpenAI, Ollama, ...) so the sample runs without an API key.</summary>
internal sealed class EchoChatClient(string name) : IChatClient
{
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, $"[{name}] you said: {messages.Last().Text}")));

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        yield return new ChatResponseUpdate(ChatRole.Assistant, (await GetResponseAsync(messages, options, cancellationToken)).Text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose() { }
}
