# 2. Dependency injection registration

**Goal:** make NLaya feel native in ASP.NET Core and the Generic Host. It should register with one
line, take its logging from the app, load models at startup instead of on the first request, and be
disposed by the container. Pair this with [step 1](01-microsoft-extensions-ai.md): both are small
and ship together.

## Shape

Put the extensions in `NLaya.Extensions.AI` (from step 1). `Microsoft.Extensions.AI` already brings
in `Microsoft.Extensions.DependencyInjection.Abstractions`, so no extra package is needed. If the
extensions should be usable without the AI package, a separate `NLaya.Extensions.DependencyInjection`
is the alternative; keep it to one package unless there's a reason.

```csharp
builder.Services.AddLaya(Laya.MultilingualModel, o => o.UseTorchSharp());          // singleton LayaAgent
builder.Services.AddKeyedLaya("english", Laya.DefaultModel, o => o.UseTorchSharp()); // several checkpoints
builder.Services.AddLayaRouter(o => o.ConfigureAgent = (_, a) => a.UseTorchSharp());  // singleton Router
builder.Services.AddChatClient(...).UseLayaGuardrail();                               // step 1, resolves ILayaPredictor
```

- **Lifetime:** singleton. Agents and the Router are thread-safe and expensive to build.
- **Logging:** take `ILoggerFactory` from the container and pass `LayaOptions.Logger` /
  `RouterOptions.Logger` when the caller hasn't set one.
- **Warm-up:** register an `IHostedService` that builds the agent, or runs `Router.Preload(...)`,
  in `StartAsync`, so the first request doesn't pay the 1–4 s model load. Make it opt-out.
- **Configuration binding:** bind `IOptions<LayaSettings>` from a `"Laya"` section for the values
  that aren't code: `Model`, `Subfolder`, `Revision`, `CacheDir`, `MaxLen`, and the Router's
  `MaxLoaded`, `Default` and `AutoTaskDetection`. The backend stays in code, because each backend
  is its own package (`UseTorchSharp` / `UseOnnx`).
- **Registered services:** `LayaAgent` (or `Router`) and `ILayaPredictor` (from step 1) resolving to
  the same instance.
- **Health check (optional):** `IHealthCheck` reporting whether the model is loaded. Use
  `Microsoft.Extensions.Diagnostics.HealthChecks` only if it doesn't bloat the package; otherwise
  leave it out.

## Tests

Build a `ServiceCollection` and resolve the services:
- Singletons are shared, keyed registrations resolve separately, and `ILayaPredictor` is the same
  object as the agent.
- Disposing the provider disposes the agent.
- Configuration binding works.
- The model load itself can use a fake `ILayaBackendFactory` (`LayaOptions.Backend`), so no model
  is needed. This needs a tokenizer; the cached multilingual one is fine, and skip when absent, like
  `TestFiles.Tokenizer`.

## Done when

A minimal API sample (`samples/NLaya.WebApi`, about 30 lines) exposes `POST /triage`, running
`Presets.Triage()` through the injected `Router`, and the README has a "Dependency injection" section.
