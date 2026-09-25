using static TorchSharp.torch;

namespace NLaya.Eval;

/// <summary>Command-line settings; see <see cref="Usage"/>.</summary>
internal sealed class EvalArgs
{
    public const string Usage = """
        NLaya.Eval: accuracy, calibration and latency of a Laya checkpoint on laya's benchmark suites.

          dotnet run -c Release --project benchmarks/NLaya.Eval -- --suite <file.jsonl> [--suite ...] [options]

        Suites come from benchmarks/export_suites.py. Options:
          --model <id|dir>       checkpoint (default convaiinnovations/laya); a Hub id is read from the HF cache
          --subfolder <name>     one checkpoint of a bundle repo, e.g. typed-decisions or multilingual
          --onnx <dir>           run an export_onnx.py export with ONNX Runtime instead of TorchSharp
          --device <d>           cpu (default), cuda, cuda:1, mps
          --dtype <t>            float32 (default), bfloat16, float16 (TorchSharp)
          --batch-size <n>       states per forward pass (default 16)
          --limit <n>            only the first n cases of each suite
          --refit <mode>         per-bucket (the harness's refit) or per-type (the fine-tuning notebook's):
                                 fit on half of each suite, report ECE on the other half
          --save-config <path>   fit temperatures on every suite given (with --refit's mode, default per-type)
                                 and write the checkpoint's rl_agent_config.json with them to <path>
          --latency <n>          time n single-question Predict calls (after a warm-up)
          --report <path>        write every number as JSON
        """;

    public List<string> Suites { get; } = [];
    public string Model { get; private set; } = Laya.DefaultModel;
    public string? Subfolder { get; private set; }
    public string? Onnx { get; private set; }
    public string Device { get; private set; } = "cpu";
    public ScalarType DType { get; private set; } = ScalarType.Float32;
    public int BatchSize { get; private set; } = 16;
    public int? Limit { get; private set; }
    public string? Refit { get; private set; }
    public string? SaveConfig { get; private set; }
    public int Latency { get; private set; }
    public string? Report { get; private set; }

    public static EvalArgs Parse(string[] args)
    {
        var o = new EvalArgs();
        for (var i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--suite": o.Suites.Add(Next()); break;
                case "--model": o.Model = Next(); break;
                case "--subfolder": o.Subfolder = Next(); break;
                case "--onnx": o.Onnx = Next(); break;
                case "--device": o.Device = Next(); break;
                case "--dtype":
                    o.DType = Next() switch
                    {
                        "float32" => ScalarType.Float32,
                        "bfloat16" => ScalarType.BFloat16,
                        "float16" => ScalarType.Float16,
                        var d => throw new ArgumentException($"--dtype {d}: use float32, bfloat16 or float16"),
                    };
                    break;
                case "--batch-size": o.BatchSize = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--limit": o.Limit = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--refit":
                    o.Refit = Next();
                    if (o.Refit is not ("per-bucket" or "per-type")) throw new ArgumentException("--refit takes per-bucket or per-type");
                    break;
                case "--save-config": o.SaveConfig = Next(); break;
                case "--latency": o.Latency = int.Parse(Next(), System.Globalization.CultureInfo.InvariantCulture); break;
                case "--report": o.Report = Next(); break;
                default: throw new ArgumentException($"unknown argument {args[i]}");
            }
        }
        if (o.Suites.Count == 0) throw new ArgumentException("give at least one --suite");
        return o;
    }
}
