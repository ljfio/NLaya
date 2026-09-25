using System.Diagnostics;

using NLaya;
using NLaya.Email;
using NLaya.Routing;
using NLaya.TorchSharp;

// One agent, straight from the Hub (the quickstart on huggingface.co/convaiinnovations/laya-multilingual).
await using (var agent = await Laya.LoadAsync("convaiinnovations/laya-multilingual", o => o.UseTorchSharp("cpu")))
{
    var questions = new Questions
    {
        ["department"] = Question.Choice("Which team should handle `body`?", new Dictionary<string, string?>
        {
            ["billing"] = "invoices, payments, refunds",
            ["technical"] = "bugs and outages",
            ["sales"] = "pricing",
        }),
        ["refund_requested"] = Question.Noul("Does the sender ask for money back?"),
    };
    var sw = Stopwatch.StartNew();
    var result = agent.Predict(new { body = "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।" }, questions);
    Console.WriteLine($"[{agent.Backend.Name}, {sw.ElapsedMilliseconds} ms] department={result.Choice("department").Choice} " +
                      $"refund={result.Noul("refund_requested").Probability}");
}

// The Router picks the checkpoint per request: English, multilingual, or typed-decisions on request.
using var router = new Router(o => o.UseTorchSharp("cpu"));
var triage = Presets.Triage();
var email = EmailCleaner.State("Charged twice",
    "Hi,\n\nI was charged twice for invoice 4411, please refund me today.\n\nThanks,\nAna\n\nSent from my iPhone\n\nOn Mon, Jun 3, 2024 at 10:00 AM Support <s@x.com> wrote:\n> old thread");

foreach (var (label, state, options) in new (string, LayaState, RouteOptions?)[]
{
    ("english email", email, null),
    ("japanese", "二重に請求されました。返金してください。", null),
    ("typed-decisions", "Customer threatens to cancel after a double charge", new RouteOptions { Checkpoint = Checkpoint.TypedDecisions }),
})
{
    var sw = Stopwatch.StartNew();
    var r = router.Predict(state, triage, options);
    Console.WriteLine($"[{r.Routing!.Checkpoint.Name()}, {sw.ElapsedMilliseconds} ms] {label}: intent={r.Choice("intent").Choice} " +
                      $"frustration={r.Score("frustration").Score} refund={r.Noul("refund_requested").Probability} — {r.Routing.Reason}");
}
