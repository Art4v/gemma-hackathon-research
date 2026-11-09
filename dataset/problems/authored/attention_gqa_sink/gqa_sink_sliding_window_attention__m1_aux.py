"""[m1 M2_second_output variant of authored/attention_gqa_sink/gqa_sink_sliding_window_attention] Sliding-window GQA attention with sink logits that also returns the per-row logsumexp over [keys, sink] as (B, S, Hq) in f32."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gqa_sink_sliding_window_attention_m1_aux',
    'batch': 4,
    'seq_len': 4096,
    'num_q_heads': 32,
    'num_kv_heads': 8,     # 4:1 query-to-kv head ratio
    'head_dim': 128,
    'window': 1024,        # key j visible to query i iff j <= i and i - j < window
    'q_chunk': 512,        # static query-chunk size used by the reference to bound memory
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    kq, kk, kv, ks = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype)
    k = jax.random.normal(kk, (B, S, Hkv, D), dtype)
    v = jax.random.normal(kv, (B, S, Hkv, D), dtype)
    # Learned per-query-head sink logits; kept in float32 like gpt-oss (cast to f32 before softmax).
    sinks = jax.random.normal(ks, (Hq,), jnp.float32)
    return (q, k, v, sinks)


def workload(q, k, v, sinks):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    G = Hq // Hkv
    W = CONFIG['window']
    C = CONFIG['q_chunk']
    neg = CONFIG['mask_value']
    assert S % C == 0 and Hq % Hkv == 0
    scale = float(D) ** -0.5

    # Group query heads by their shared kv head: (B, S, Hkv, G, D)
    qg = q.reshape(B, S, Hkv, G, D)
    sink = sinks.astype(jnp.float32).reshape(1, Hkv, G, 1)  # broadcasts over (B, Hkv, G, C)

    outs, lses = [], []
    for qs in range(0, S, C):
        qe = qs + C
        ks = max(0, qs - W + 1)          # earliest key any query in this chunk can see
        qc = qg[:, qs:qe]                # (B, C, Hkv, G, D)
        kc = k[:, ks:qe]                 # (B, Kc, Hkv, D)
        vc = v[:, ks:qe]                 # (B, Kc, Hkv, D)

        s = jnp.einsum('bqhgd,bkhd->bhgqk', qc, kc,
                       preferred_element_type=jnp.float32) * scale   # (B, Hkv, G, C, Kc)

        qi = jnp.arange(qs, qe)[:, None]
        kj = jnp.arange(ks, qe)[None, :]
        mask = (kj <= qi) & ((qi - kj) < W)                            # (C, Kc)
        s = jnp.where(mask, s, neg)

        # Softmax over [real keys, sink]: the sink joins the max and the denominator only.
        m = jnp.maximum(jnp.max(s, axis=-1), sink)                     # (B, Hkv, G, C)
        p = jnp.exp(s - m[..., None])                                  # (B, Hkv, G, C, Kc)
        l = jnp.sum(p, axis=-1) + jnp.exp(sink - m)                    # (B, Hkv, G, C)

        o = jnp.einsum('bhgqk,bkhd->bhgqd', p.astype(vc.dtype), vc,
                       preferred_element_type=jnp.float32)            # (B, Hkv, G, C, D)
        outs.append(o / l[..., None])
        lses.append(m + jnp.log(l))                                    # (B, Hkv, G, C)

    out = jnp.concatenate(outs, axis=3)                                # (B, Hkv, G, S, D)
    out = out.transpose(0, 3, 1, 2, 4).reshape(B, S, Hq, D)
    lse = jnp.concatenate(lses, axis=3)                                # (B, Hkv, G, S)
    lse = lse.transpose(0, 3, 1, 2).reshape(B, S, Hq)
    return out.astype(q.dtype), lse
