namespace NLaya.TorchSharp;

/// <summary>One tensor in a <c>.safetensors</c> header: dtype, shape and byte range in the data section.</summary>
internal sealed record SafeTensorEntry(string Name, string DType, long[] Shape, long Start, long End);
