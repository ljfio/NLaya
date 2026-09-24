using System.Runtime.CompilerServices;

using Microsoft.Extensions.AI;

namespace NLaya.Tests.ExtensionsAI;

/// <summary>A chat client that replies "&lt;name&gt;: &lt;last message&gt;" and counts its calls.</summary>
internal sealed class EchoChatClient(string name = "echo") : IChatClient
{
    public int Calls { get; private set; }

    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        Calls++;
        return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, $"{name}: {messages.Last().Text}")));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Calls++;
        await Task.Yield();
        yield return new ChatResponseUpdate(ChatRole.Assistant, $"{name}: ");
        yield return new ChatResponseUpdate(ChatRole.Assistant, messages.Last().Text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() { }
}
