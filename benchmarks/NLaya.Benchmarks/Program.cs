using BenchmarkDotNet.Running;

using NLaya.Benchmarks;

// dotnet run -c Release --project benchmarks/NLaya.Benchmarks -- --filter '*'
// NLAYA_DEVICE (cpu, cuda, mps) and NLAYA_DTYPE (float32, bfloat16, float16) pick the device and precision.
BenchmarkSwitcher.FromAssembly(typeof(PredictBenchmarks).Assembly).Run(args);
