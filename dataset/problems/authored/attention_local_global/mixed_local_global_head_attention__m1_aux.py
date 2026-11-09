"""[m1 M2_second_output variant of authored/attention_local_global/mixed_local_global_head_attention] Mixed local/global soft-capped attention that also returns the per-token absmax of the attention output over all heads (B, S), the scale for quantizing the output-projection input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mixed_local_global_head_attention_m1_aux',
    'batch': 2,
    'seq_len': 4096,
    'num_heads': 16,
    'head_dim': 128,
    'window': 1024,
    'softcap': 50.0,
    'q_chunk': 512,
}


def create_inputs(dtype=jnp.float32):
    B, S, H, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim']
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (B, S, H, D), dtype)
    k = jax.random.normal(kk, (B, S, H, D), dtype)
    v = jax.random.normal(kv, (B, S, H, D), dtype)
    return q, k, v


def workload(q, k, v):
    B, S, H, D = q.shape
    Cq = CONFIG['q_chunk']
    nC = S // Cq
    scale = D ** -0.5
    cap = CONFIG['softcap']
    # Even heads: local sliding window; odd heads: global (window = S).
    head_window = jnp.where(jnp.arange(H) % 2 == 0, CONFIG['window'], S).astype(jnp.int32)
    kpos = jnp.arange(S, dtype=jnp.int32)

    q_chunks = q.reshape(B, nC, Cq, H, D).transpose(1, 0, 2, 3, 4)  # (nC,B,Cq,H,D)

    def body(args):
        qc, c = args
        s = jnp.einsum('bqhd,bkhd->bhqk', qc, k,
                       preferred_element_type=jnp.float32) * scale
        s = cap * jnp.tanh(s / cap)
        qpos = c * Cq + jnp.arange(Cq, dtype=jnp.int32)
        diff = qpos[:, None] - kpos[None, :]  # (Cq,S)
        mask = (diff[None] >= 0) & (diff[None] < head_window[:, None, None])  # (H,Cq,S)
        s = jnp.where(mask[None], s, jnp.float32(-1e30))
        m = jnp.max(s, axis=-1, keepdims=True)
        p = jnp.exp(s - m)
        l = jnp.sum(p, axis=-1, keepdims=True)
        p = p / l
        o = jnp.einsum('bhqk,bkhd->bqhd', p.astype(v.dtype), v,
                       preferred_element_type=jnp.float32)
        return o.astype(q.dtype)

    out = lax.map(body, (q_chunks, jnp.arange(nC, dtype=jnp.int32)))  # (nC,B,Cq,H,D)
    out = out.transpose(1, 0, 2, 3, 4).reshape(B, S, H, D)
    # Per-token absmax over every head's output, in f32.
    amax = jnp.max(jnp.abs(out.astype(jnp.float32)), axis=(2, 3))  # (B,S)
    return out, amax
