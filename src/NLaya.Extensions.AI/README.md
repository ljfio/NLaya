# NLaya.Extensions.AI

[NLaya](https://www.nuget.org/packages/NLaya) for `Microsoft.Extensions.AI` and dependency injection:

- **`UseLayaGuardrail`**: `IChatClient` middleware that screens each user turn (with `Presets.Guard()`
  by default) and throws, refuses or annotates.
- **`LayaRouterChatClient`**: sends each request to one of several chat clients, chosen by a Laya
  `choice` question, with confidence-gated fallback.
- **`LayaTools`**: Laya as an `AIFunction` tool (for example `LayaTools.Triage(agent)`).
- **`AddLaya` / `AddKeyedLaya` / `AddLayaRouter`**: singleton registration, configuration from the
  `"Laya"` section, logging from the container, and model warm-up at host start.

```csharp
builder.Services.AddLaya(o => o.UseTorchSharp());
builder.Services.AddChatClient(innerClient)
    .UseLayaGuardrail(o => o.Thresholds["harm_severity"] = 2);
```

NLaya is a .NET port of [laya](https://github.com/NandhaKishorM/laya) by Convai Innovations / NandhaKishorM (Apache-2.0). The Laya model weights are theirs, published on Hugging Face under [convaiinnovations](https://huggingface.co/convaiinnovations). Source, docs and samples: https://github.com/ljfio/NLaya.
