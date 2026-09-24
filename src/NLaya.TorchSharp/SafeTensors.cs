using System.Buffers;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace NLaya.TorchSharp;

/// <summary>Reads a <c>.safetensors</c> file (8-byte header length, JSON header, raw little-endian data).</summary>
internal static class SafeTensors
{
    public sealed record Entry(string Name, string DType, long[] Shape, long Start, long End);

    public static IReadOnlyList<Entry> ReadHeader(string path, out long dataOffset)
    {
        using var fs = File.OpenRead(path);
        Span<byte> lenBytes = stackalloc byte[8];
        fs.ReadExactly(lenBytes);
        var headerLen = BitConverter.ToInt64(lenBytes);
        if (headerLen <= 0 || headerLen > 100_000_000) throw new InvalidDataException($"{path} is not a safetensors file");
        var header = new byte[headerLen];
        fs.ReadExactly(header);
        dataOffset = 8 + headerLen;
        using var doc = JsonDocument.Parse(header);
        var entries = new List<Entry>();
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            if (p.Name == "__metadata__") continue;
            var offs = p.Value.GetProperty("data_offsets");
            entries.Add(new Entry(p.Name, p.Value.GetProperty("dtype").GetString()!,
                p.Value.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray(),
                offs[0].GetInt64(), offs[1].GetInt64()));
        }
        return entries;
    }

    /// <summary>Load every tensor, converted to <paramref name="dtype"/> on <paramref name="device"/>.</summary>
    public static Dictionary<string, Tensor> Load(string path, ScalarType dtype, Device device)
    {
        var entries = ReadHeader(path, out var dataOffset);
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
        foreach (var e in entries.OrderBy(e => e.Start))
        {
            var src = e.DType switch
            {
                "F16" => ScalarType.Float16,
                "BF16" => ScalarType.BFloat16,
                "F32" => ScalarType.Float32,
                "F64" => ScalarType.Float64,
                "I64" => ScalarType.Int64,
                "I32" => ScalarType.Int32,
                "BOOL" => ScalarType.Bool,
                _ => throw new NotSupportedException($"safetensors dtype {e.DType} ({e.Name})"),
            };
            var len = checked((int)(e.End - e.Start));
            var buf = ArrayPool<byte>.Shared.Rent(Math.Max(1, len));
            try
            {
                fs.Position = dataOffset + e.Start;
                fs.ReadExactly(buf, 0, len);
                using var raw = empty(e.Shape, src);
                raw.bytes = buf.AsSpan(0, len);
                var isFloat = src is ScalarType.Float16 or ScalarType.BFloat16 or ScalarType.Float32 or ScalarType.Float64;
                result[e.Name] = raw.to(isFloat ? dtype : src, device).DetachFromDisposeScope();
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }
        }
        return result;
    }
}
