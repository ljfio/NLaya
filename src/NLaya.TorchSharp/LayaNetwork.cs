using NLaya.Config;

using static TorchSharp.torch;

using F = TorchSharp.torch.nn.functional;

namespace NLaya.TorchSharp;

/// <summary>
/// The Laya network, inference only: a ModernBERT encoder (transformers' <c>ModernBertModel</c>,
/// padded SDPA path) followed by the decision head of <c>laya.common.DecisionModel</c>. Weights are
/// used by their checkpoint names, so no module tree is needed.
/// </summary>
internal sealed class LayaNetwork : IDisposable
{
    private readonly Dictionary<string, Tensor> _w;
    private readonly ModernBertConfig _cfg;
    private readonly int _headLayers;
    private readonly int _headHeads;
    private readonly ScalarType _dtype;
    private readonly Dictionary<(double Theta, long Len), (Tensor Cos, Tensor Sin)> _rope = new();
    private readonly object _ropeLock = new();

    public Device Device { get; }

    public LayaNetwork(Dictionary<string, Tensor> weights, ModernBertConfig cfg, int headLayers, ScalarType dtype, Device device)
    {
        _w = weights;
        _cfg = cfg;
        _dtype = dtype;
        Device = device;
        _headLayers = Enumerable.Range(0, 64).TakeWhile(i => _w.ContainsKey($"head.layers.{i}.self_attn.in_proj_weight")).Count();
        if (_headLayers != headLayers)
            throw new InvalidDataException($"checkpoint has {_headLayers} head layers but rl_agent_config.json says {headLayers}");
        _headHeads = Math.Max(1, cfg.HiddenSize / 64);
        Verify();
    }

    private void Verify()
    {
        string[] required = ["encoder.embeddings.tok_embeddings.weight", "encoder.embeddings.norm.weight", "encoder.final_norm.weight",
            "type_emb.weight", "scorer.0.weight", "scorer.1.weight", "scorer.3.weight", "act_head.0.weight", "act_head.2.weight"];
        var missing = required.Concat(Enumerable.Range(0, _cfg.NumHiddenLayers).SelectMany(i => new[]
            {
                $"encoder.layers.{i}.attn.Wqkv.weight", $"encoder.layers.{i}.attn.Wo.weight",
                $"encoder.layers.{i}.mlp.Wi.weight", $"encoder.layers.{i}.mlp.Wo.weight", $"encoder.layers.{i}.mlp_norm.weight",
            }))
            .Where(k => !_w.ContainsKey(k)).ToList();
        if (missing.Count > 0)
            throw new InvalidDataException(
                $"checkpoint does not match encoder/config.json: missing {string.Join(", ", missing.Take(5))}{(missing.Count > 5 ? ", ..." : "")}");
        var emb = _w["encoder.embeddings.tok_embeddings.weight"];
        if (emb.shape[1] != _cfg.HiddenSize)
            throw new InvalidDataException($"embedding width {emb.shape[1]} != hidden_size {_cfg.HiddenSize}");
    }

    private Tensor W(string name) => _w[name];
    private Tensor? B(string name) => _w.TryGetValue(name, out var t) ? t : null;

    /// <summary>
    /// input_ids / attention_mask: [B, L] int64; marker_pos [B, K] int64; marker_mask [B, K] bool;
    /// qtype [B] int64. Returns logits [B, K] and act_logits [B, A], float32.
    /// </summary>
    public (Tensor Logits, Tensor ActLogits) Forward(Tensor inputIds, Tensor attentionMask, Tensor markerPos, Tensor markerMask, Tensor qtype)
    {
        var h = Encode(inputIds, attentionMask);
        return Head(h, attentionMask, markerPos, markerMask, qtype);
    }

    public Tensor Encode(Tensor inputIds, Tensor attentionMask)
    {
        var (bsz, len) = (inputIds.shape[0], inputIds.shape[1]);
        var d = _cfg.HiddenSize;
        var eps = _cfg.NormEps;

        var h = W("encoder.embeddings.tok_embeddings.weight").index_select(0, inputIds.reshape(-1)).reshape(bsz, len, d);
        h = F.layer_norm(h, [d], W("encoder.embeddings.norm.weight"), B("encoder.embeddings.norm.bias"), eps);

        // Additive masks: padding keys everywhere; sliding layers also drop keys beyond the window.
        var minVal = _dtype == ScalarType.Float32 ? float.MinValue : -65504f;
        var keyOk = attentionMask.to(ScalarType.Bool).reshape(bsz, 1, 1, len);
        var globalMask = zeros([bsz, 1, 1, len], _dtype, Device).masked_fill(keyOk.logical_not(), minVal).expand(bsz, 1, len, len);
        var pos = arange(len, device: Device);
        var dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs();
        var outside = dist.gt(_cfg.LocalAttention / 2).reshape(1, 1, len, len);
        var slidingMask = globalMask.masked_fill(outside, minVal);

        for (var i = 0; i < _cfg.NumHiddenLayers; i++)
        {
            var p = $"encoder.layers.{i}.";
            var isGlobal = _cfg.GlobalLayers[i];
            var x = _w.ContainsKey(p + "attn_norm.weight") ? F.layer_norm(h, [d], W(p + "attn_norm.weight"), B(p + "attn_norm.bias"), eps) : h;
            h = h + Attention(x, p, isGlobal ? globalMask : slidingMask, isGlobal ? _cfg.GlobalRopeTheta : _cfg.LocalRopeTheta, len);
            var m = F.layer_norm(h, [d], W(p + "mlp_norm.weight"), B(p + "mlp_norm.bias"), eps);
            var wi = F.linear(m, W(p + "mlp.Wi.weight"), B(p + "mlp.Wi.bias"));
            var parts = wi.chunk(2, -1);
            h = h + F.linear(Act(parts[0]) * parts[1], W(p + "mlp.Wo.weight"), B(p + "mlp.Wo.bias"));
        }
        return F.layer_norm(h, [d], W("encoder.final_norm.weight"), B("encoder.final_norm.bias"), eps);
    }

    private Tensor Act(Tensor x) => _cfg.HiddenActivation switch
    {
        "gelu" => F.gelu(x),
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => F.gelu(x, global::TorchSharp.Modules.GELU.Approximate.tanh),
        "relu" => F.relu(x),
        "silu" or "swish" => F.silu(x),
        _ => throw new NotSupportedException($"hidden_activation '{_cfg.HiddenActivation}'"),
    };

    private Tensor Attention(Tensor x, string p, Tensor mask, double theta, long len)
    {
        var (bsz, heads, dh) = (x.shape[0], _cfg.NumAttentionHeads, _cfg.HeadDim);
        var qkv = F.linear(x, W(p + "attn.Wqkv.weight"), B(p + "attn.Wqkv.bias")).view(bsz, len, 3, heads, dh);
        var q = qkv.select(2, 0).transpose(1, 2);
        var k = qkv.select(2, 1).transpose(1, 2);
        var v = qkv.select(2, 2).transpose(1, 2);
        var (cos, sin) = Rope(theta, len, dh);
        q = q * cos + RotateHalf(q) * sin;
        k = k * cos + RotateHalf(k) * sin;
        var o = F.scaled_dot_product_attention(q, k, v, mask, 0.0, false);
        o = o.transpose(1, 2).reshape(bsz, len, heads * dh);
        return F.linear(o, W(p + "attn.Wo.weight"), B(p + "attn.Wo.bias"));
    }

    private static Tensor RotateHalf(Tensor x)
    {
        var halves = x.chunk(2, -1);
        return cat([halves[1].neg(), halves[0]], -1);
    }

    private (Tensor Cos, Tensor Sin) Rope(double theta, long len, int dim)
    {
        lock (_ropeLock)
        {
            if (_rope.TryGetValue((theta, len), out var cached)) return cached;
            using var scope = NewDisposeScope();
            // Same float32 arithmetic as transformers' default rotary embedding.
            var invFreq = 1.0f / pow(tensor(theta, ScalarType.Float32), arange(0, dim, 2, ScalarType.Int64).to(ScalarType.Float32) / dim);
            var t = arange(len, ScalarType.Int64).to(ScalarType.Float32);
            var freqs = outer(t, invFreq);
            var emb = cat([freqs, freqs], -1);
            var cos = emb.cos().to(_dtype, Device).reshape(1, 1, len, dim).MoveToOuterDisposeScope();
            var sin = emb.sin().to(_dtype, Device).reshape(1, 1, len, dim).MoveToOuterDisposeScope();
            cos.DetachFromDisposeScope();
            sin.DetachFromDisposeScope();
            if (_rope.Count > 32)
            {
                foreach (var (c, s) in _rope.Values) { c.Dispose(); s.Dispose(); }
                _rope.Clear();
            }
            _rope[(theta, len)] = (cos, sin);
            return (cos, sin);
        }
    }

    private (Tensor Logits, Tensor ActLogits) Head(Tensor h, Tensor attentionMask, Tensor markerPos, Tensor markerMask, Tensor qtype)
    {
        var (bsz, len, d) = (h.shape[0], h.shape[1], h.shape[2]);
        h = h + W("type_emb.weight").index_select(0, qtype).unsqueeze(1);

        var minVal = _dtype == ScalarType.Float32 ? float.MinValue : -65504f;
        var padMask = zeros([bsz, 1, 1, len], _dtype, Device)
            .masked_fill(attentionMask.to(ScalarType.Bool).logical_not().reshape(bsz, 1, 1, len), minVal);
        var heads = _headHeads;
        var dh = d / heads;
        for (var i = 0; i < _headLayers; i++)
        {
            // nn.TransformerEncoderLayer(norm_first=True, activation=relu), eps 1e-5.
            var p = $"head.layers.{i}.";
            var x = F.layer_norm(h, [d], W(p + "norm1.weight"), B(p + "norm1.bias"), 1e-5);
            var qkv = F.linear(x, W(p + "self_attn.in_proj_weight"), B(p + "self_attn.in_proj_bias")).view(bsz, len, 3, heads, dh);
            var q = qkv.select(2, 0).transpose(1, 2);
            var k = qkv.select(2, 1).transpose(1, 2);
            var v = qkv.select(2, 2).transpose(1, 2);
            var a = F.scaled_dot_product_attention(q, k, v, padMask, 0.0, false).transpose(1, 2).reshape(bsz, len, d);
            h = h + F.linear(a, W(p + "self_attn.out_proj.weight"), B(p + "self_attn.out_proj.bias"));
            var y = F.layer_norm(h, [d], W(p + "norm2.weight"), B(p + "norm2.bias"), 1e-5);
            y = F.linear(F.relu(F.linear(y, W(p + "linear1.weight"), B(p + "linear1.bias"))), W(p + "linear2.weight"), B(p + "linear2.bias"));
            h = h + y;
        }

        var idx = markerPos.clamp_min(0).unsqueeze(-1).expand(-1, -1, d);
        var m = h.gather(1, idx);
        m = F.layer_norm(m, [d], W("scorer.0.weight"), B("scorer.0.bias"), 1e-5);
        m = F.gelu(F.linear(m, W("scorer.1.weight"), B("scorer.1.bias")));
        var logits = F.linear(m, W("scorer.3.weight"), B("scorer.3.bias")).squeeze(-1).to(ScalarType.Float32);
        logits = logits.masked_fill(markerMask.logical_not(), -1e4f);

        var prob = logits.softmax(-1);
        var kCount = markerMask.sum(-1).clamp_min(2).to(ScalarType.Float32);
        var ent = -(prob * prob.clamp_min(1e-9f).log()).sum(-1) / kCount.log();
        Tensor top1, top2;
        if (prob.shape[1] >= 2)
        {
            var (vals, _) = prob.topk(2, -1);
            top1 = vals.select(1, 0);
            top2 = vals.select(1, 1);
        }
        else
        {
            // One option: softmax is 1.0; pad the missing runner-up with 0 (fully decided).
            top1 = prob.select(1, 0);
            top2 = zeros_like(top1);
        }
        var feats = stack([top1, top1 - top2, ent, kCount / 255.0f], -1);
        var pooled = h.select(1, 0).to(ScalarType.Float32);
        var z = cat([pooled, feats], -1);
        if (_dtype != ScalarType.Float32) z = z.to(_dtype);
        z = F.gelu(F.linear(z, W("act_head.0.weight"), B("act_head.0.bias")));
        var act = F.linear(z, W("act_head.2.weight"), B("act_head.2.bias")).to(ScalarType.Float32);
        return (logits, act);
    }

    public void Dispose()
    {
        foreach (var t in _w.Values) t.Dispose();
        _w.Clear();
        foreach (var (c, s) in _rope.Values) { c.Dispose(); s.Dispose(); }
        _rope.Clear();
    }
}
