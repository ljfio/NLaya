using Microsoft.Extensions.AI;

namespace NLaya.Extensions.AI;

/// <summary>Which text of a chat request Laya reads. Port of the message branch of Python's <c>_extract_text</c>.</summary>
internal static class ChatText
{
    /// <summary>The newest user message, else the newest message of any role.</summary>
    public static string LatestUserText(IEnumerable<ChatMessage> messages)
    {
        var list = messages as IList<ChatMessage> ?? messages.ToList();
        if (list.Count == 0) return "";
        for (var i = list.Count - 1; i >= 0; i--)
            if (list[i].Role == ChatRole.User) return list[i].Text;
        return list[^1].Text;
    }
}
