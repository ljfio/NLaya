// Native AOT smoke test. Publish with `dotnet publish samples/NLaya.AotSmoke -c Release -r <rid>`
// (IL trim/AOT warnings fail the publish), then run the binary:
//
//   NLaya.AotSmoke                      model-free checks (questions, JSON states, language, email)
//   NLaya.AotSmoke --require-tokenizer  also fail when the multilingual tokenizer isn't cached; with it,
//                                       tokenize, decode and bind DI settings against a fake backend
//   NLaya.AotSmoke --onnx <dir>         also run a real prediction on an export_onnx.py output
using System.Text.Json.Nodes;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

using NLaya;
using NLaya.AotSmoke;
using NLaya.Email;
using NLaya.Extensions.AI;
using NLaya.Hub;
using NLaya.Lang;
using NLaya.Onnx;

var requireTokenizer = args.Contains("--require-tokenizer");
var onnxAt = Array.IndexOf(args, "--onnx");
var onnxDir = onnxAt >= 0 && onnxAt + 1 < args.Length ? args[onnxAt + 1] : null;
var failures = 0;

void Check(string name, bool ok, string? detail = null)
{
    Console.WriteLine($"{(ok ? "ok  " : "FAIL")} {name}{(detail is null ? "" : ": " + detail)}");
    if (!ok) failures++;
}

// Questions: the Python dict shape round-trips, and presets load from embedded JSON.
var parsed = Questions.Parse("""{"refund": {"type": "noul", "instructions": "Refund?", "labels": {"false": "no", "true": "yes"}}}""");
Check("questions round-trip", parsed.ToJson().ToJsonString() == Questions.FromJson(parsed.ToJson()).ToJson().ToJsonString());
var triage = Presets.Triage();
Check("presets", triage.ContainsKey("intent"), $"{triage.Count} triage questions");

// States: source-generated POCOs, CLR JSON values, conversations.
var ticket = new Ticket("Invoice 4411", "We were billed twice for March.", ["billing"]);
var state = LayaState.From(ticket, SmokeJsonContext.Default.Ticket);
Check("POCO state", state.Serialize() == """{"Subject": "Invoice 4411", "Body": "We were billed twice for March.", "Tags": ["billing"]}""", state.Serialize());
var clr = LayaState.FromJson(new JsonObject { ["n"] = 3, ["f"] = 2.0, ["l"] = 9_000_000_000L, ["g"] = Guid.Empty, ["c"] = 'é' });
Check("CLR JSON values", clr.Serialize() == """{"n": 3, "f": 2.0, "l": 9000000000, "g": "00000000-0000-0000-0000-000000000000", "c": "é"}""", clr.Serialize());
Check("conversation", LayaState.Conversation(["hi", state]).IsConversation);

// Language detection and email cleaning read embedded tables.
Check("language", !LanguageDetector.IsEnglish("Mein Konto wurde zweimal belastet, bitte erstatten Sie es."));
Check("email", EmailCleaner.State("Refund", "Please refund me.", "a@example.com")["body"] is not null);

// Tokenizer, prompt layout, decoding and DI with source-generated configuration binding.
if (HfCache.Snapshot(Laya.MultilingualModel) is not null)
{
    var config = new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
    {
        ["Laya:Model"] = Laya.MultilingualModel,
        ["Laya:Warmup"] = "false",
        ["Laya:Preload:0"] = "multilingual",
    }).Build();
    var services = new ServiceCollection().AddSingleton<IConfiguration>(config);
    services.AddLaya(o => o.Backend = new FakeBackendFactory());
    await using var sp = services.BuildServiceProvider();
    var settings = sp.GetRequiredService<IOptionsMonitor<LayaSettings>>().Get(Options.DefaultName);
    Check("DI settings bound", settings is { Warmup: false, Preload: ["multilingual"] });
    var agent = sp.GetRequiredService<LayaAgent>();
    Check("DI model from config", agent.ModelId == Laya.MultilingualModel, agent.ModelId);
    var result = agent.Predict(state, triage);
    Check("fake-backend predict", result.Answers.Count == triage.Count && result.Usage.InputTokens > 0, $"{result.Usage.InputTokens} tokens");
    Check("result JSON", JsonNode.Parse(result.ToJsonString())?["answers"]?["intent"] is not null);
}
else
{
    Check("tokenizer cached", !requireTokenizer, "skipped; run: " + HfCache.DownloadCommand(Laya.MultilingualModel));
}

// A real forward pass through ONNX Runtime.
if (onnxDir is not null)
{
    using var agent = Laya.Load(onnxDir, o => o.UseOnnx(onnxDir));
    var result = agent.Predict("We were billed twice for March, please refund it.", triage);
    Check("onnx predict", result.Choice("intent").Choice == "refund", result.ToJsonString());
}

Console.WriteLine(failures == 0 ? "AOT smoke passed" : $"{failures} check(s) failed");
return failures == 0 ? 0 : 1;
