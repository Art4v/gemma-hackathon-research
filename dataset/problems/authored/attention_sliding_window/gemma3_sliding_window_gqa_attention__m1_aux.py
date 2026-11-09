"""[m1 M2_second_output variant of authored/attention_sliding_window/gemma3_sliding_window_gqa_attention] Gemma-3 sliding-window GQA attention that also returns the per-row softmax logsumexp of the banded scores as (B, S, Hq) in f32."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'gemma3_sliding_window_gqa_attention_m1_aux',
    'batch': 1,
    'seq_len': 8192,
    'num_q_heads': 16,
    'num_kv_heads': 8,
    'head_dim': 256,
    'window': 1024,          # tokens each query may see, including itself (Gemma 3 local layers)
    'scale': 256 ** -0.5,    # query_pre_attn_scalar = head_dim
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype)
    k = jax.random.normal(kk, (B, S, Hkv, D), dtype)
    v = jax.random.normal(kv, (B, S, Hkv, D), dtype)
    return (q, k, v)


def workload(q, k, v):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    W = CONFIG['window']
    scale = CONFIG['scale']
    group = Hq // Hkv
    nb = S // W

    # Block the sequence into windows of W. Query block i may only see key
    # blocks i-1 and i (a 2W-wide band), so pad one zero block in front of K/V.
    qb = q.reshape(B, nb, W, Hq, D)
    kb = k.reshape(B, nb, W, Hkv, D)
    vb = v.reshape(B, nb, W, Hkv, D)
    zero_k = jnp.zeros((B, 1, W, Hkv, D), k.dtype)
    zero_v = jnp.zeros((B, 1, W, Hkv, D), v.dtype)
    k_pad = jnp.concatenate([zero_k, kb], axis=1)       # (B, nb+1, W, Hkv, D)
    v_pad = jnp.concatenate([zero_v, vb], axis=1)
    k_band = jnp.concatenate([k_pad[:, :-1], k_pad[:, 1:]], axis=2)  # (B, nb, 2W, Hkv, D)
    v_band = jnp.concatenate([v_pad[:, :-1], v_pad[:, 1:]], axis=2)

    # GQA: expand KV heads to query heads.
    k_band = jnp.repeat(k_band, group, axis=3)          # (B, nb, 2W, Hq, D)
    v_band = jnp.repeat(v_band, group, axis=3)

    # Scores in f32 with pre-scaled queries.
    qs = (qb.astype(jnp.float32) * scale).astype(q.dtype)
    s = jnp.einsum('bnrhd,bnchd->bnhrc', qs, k_band,
                   preferred_element_type=jnp.float32)  # (B, nb, Hq, W, 2W)

    # Band mask: key index c (0..2W-1) relative to block start of block i-1;
    # query row r in block i. Allowed iff 0 <= q - key < W  <=>  r < c <= W + r.
    r = lax.broadcasted_iota(jnp.int32, (W, 2 * W), 0)
    c = lax.broadcasted_iota(jnp.int32, (W, 2 * W), 1)
    allowed = (c > r) & (c <= r + W)
    s = jnp.where(allowed, s, jnp.float32(-1e30))

    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    p = p / l                                            # f32 probabilities
    lse = (m + jnp.log(l))[..., 0]                       # (B, nb, Hq, W)

    o = jnp.einsum('bnhrc,bnchd->bnrhd', p.astype(v.dtype), v_band,
                   preferred_element_type=jnp.float32)  # (B, nb, W, Hq, D)
    lse = lse.transpose(0, 1, 3, 2).reshape(B, S, Hq)
    return o.reshape(B, S, Hq, D).astype(q.dtype), lse
