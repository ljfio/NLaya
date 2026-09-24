using System.Diagnostics;
using NLaya;
using NLaya.TorchSharp;

// Mirrors the quickstart on https://huggingface.co/convaiinnovations/laya-multilingual
await using var agent = await Laya.LoadAsync("convaiinnovations/laya-multilingual", o => o.UseTorchSharp("cpu"));

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
Console.WriteLine($"{sw.ElapsedMilliseconds} ms on {agent.Backend.Name}");
Console.WriteLine(result.Choice("department").Choice);
Console.WriteLine(result.ToJsonString(indented: true));
