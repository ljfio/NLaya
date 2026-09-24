using NLaya;
using NLaya.Routing;
using NLaya.TorchSharp;

// Needs `hf download convaiinnovations/laya`. The "Laya" section of appsettings.json configures the router.
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddLayaRouter(o => o.ConfigureAgent = (_, a) => a.UseTorchSharp("cpu"));

var app = builder.Build();
var triage = Presets.Triage();

// curl -X POST localhost:5000/triage -H 'Content-Type: application/json' -d '{"message":"I was charged twice, refund me"}'
app.MapPost("/triage", async (TriageRequest request, Router router, CancellationToken ct) =>
{
    var result = await router.PredictAsync(LayaState.From(new { message = request.Message }), triage, ct: ct);
    return Results.Text(result.ToJsonString(), "application/json");
});

app.Run();

internal sealed record TriageRequest(string Message);
