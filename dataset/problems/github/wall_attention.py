"""[bf16 variant] Wall attention: per-channel log-gated q.k logits (cumsum gates) + causal mask + scalar gate bias + sink-softmax + PV."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'wall_attention',
    'batch': 8,
    'seq_len': 1024,
    'num_q_heads': 8,
    'num_kv_heads': 2,
    'head_dim': 128,
    'gate_offset': 4.0,
    'block_q': 128,
}

RCP_LN2 = 1.0 / math.log(2.0)


def create_inputs(dtype=jnp.bfloat16):
    B, T = CONFIG['batch'], CONFIG['seq_len']
    HQ, H, D = CONFIG['num_q_heads'], CONFIG['num_kv_heads'], CONFIG['head_dim']
    off = CONFIG['gate_offset']
    kq, kk, kv, kg, ks = jax.random.split(jax.random.key(0), 5)
    q = jax.random.normal(kq, (B, T, HQ, D), dtype)
    k = jax.random.normal(kk, (B, T, H, D), dtype)
    v = jax.random.normal(kv, (B, T, H, D), dtype)
    g = jax.nn.log_sigmoid(jax.random.normal(kg, (B, T, HQ, D), dtype) + off)
    g_scalar = jax.nn.log_sigmoid(jax.random.normal(ks, (B, T, HQ), dtype) + off)
    sink_bias = jnp.zeros((HQ,), dtype)
    return q, k, v, g, g_scalar, sink_bias


def workload(q, k, v, g, g_scalar, sink_bias):
    B, T, HQ, D = q.shape
    G = HQ // k.shape[2]
    bq = min(CONFIG['block_q'], T)
    assert T % bq == 0, 'seq_len must be a multiple of block_q'
    n_blocks = T // bq
    scale_l2 = (D ** -0.5) * RCP_LN2

    q = q.astype(jnp.float32)
    P = jnp.cumsum(g.astype(jnp.float32), axis=1) * RCP_LN2          # (B,T,HQ,D)
    c = jnp.cumsum(g_scalar.astype(jnp.float32), axis=1) * RCP_LN2   # (B,T,HQ)
    k_exp = jnp.repeat(k.astype(jnp.float32), G, axis=2)             # (B,T,HQ,D)
    v_exp = jnp.repeat(v.astype(jnp.float32), G, axis=2)             # (B,T,HQ,D)
    sink_l2 = sink_bias.astype(jnp.float32) * RCP_LN2                # (HQ,)
    j_idx = jnp.arange(T)[None, :]

    def per_batch(args):
        qb, Pb, cb, kb, vb = args  # (T,HQ,D) x4, cb (T,HQ)

        def per_block(blk):
            start = blk * bq
            qs = lax.dynamic_slice_in_dim(qb, start, bq, 0)
            Ps = lax.dynamic_slice_in_dim(Pb, start, bq, 0)
            cs = lax.dynamic_slice_in_dim(cb, start, bq, 0)
            # ponytail: literal exp2(P_i - P_j) per channel, O(bq*T*HQ*D) memory per block; stable for any T.
            diff = Ps[:, None] - Pb[None]                                 # (bq,T,HQ,D)
            s = jnp.sum(qs[:, None] * kb[None] * jnp.exp2(diff), axis=-1)  # (bq,T,HQ)
            s = jnp.transpose(s, (2, 0, 1))                               # (HQ,bq,T)
            i_idx = start + jnp.arange(bq)[:, None]
            s = jnp.where(j_idx <= i_idx, s, -jnp.inf) * scale_l2
            s = s + cs.T[:, :, None] - cb.T[:, None, :]
            m = jnp.max(s, axis=-1, keepdims=True)
            m = jnp.where(jnp.isfinite(m), m, 0.0)
            p = jnp.exp2(s - m)
            den = jnp.sum(p, axis=-1) + jnp.exp2(sink_l2[:, None] - m[..., 0])
            w = p / den[..., None]
            return jnp.einsum('hij,jhd->ihd', w, vb, precision=lax.Precision.HIGHEST)  # (bq,HQ,D)

        out = lax.map(per_block, jnp.arange(n_blocks))  # (nb,bq,HQ,D)
        return out.reshape(T, HQ, D)

    return lax.map(per_batch, (q, P, c, k_exp, v_exp))  # (B,T,HQ,D)
