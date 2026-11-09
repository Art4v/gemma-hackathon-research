"""[bf16 variant] Mixture-of-attention-heads gate: QKV proj -> raw scores -> scores@V -> gate softmax over Tq -> gated head sum -> @inputs -> out proj."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mixture_of_attention_heads_gate',
    'batch': 8,
    'seq_q': 1024,
    'seq_k': 1024,
    'd_model': 2048,
    'n_heads': 8,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    B, Tq, Tk = CONFIG['batch'], CONFIG['seq_q'], CONFIG['seq_k']
    D, N = CONFIG['d_model'], CONFIG['n_heads']
    hd = D // N
    ks = jax.random.split(jax.random.key(0), 13)
    query = jax.random.uniform(ks[0], (B, Tq, D), dtype)
    key = jax.random.uniform(ks[1], (B, Tk, D), dtype)
    inputs = jax.random.uniform(ks[2], (B, Tk, D), dtype)
    wq = jax.random.normal(ks[3], (D, D), dtype) * 0.02
    bq = jax.random.normal(ks[4], (D,), dtype) * 0.02
    wk = jax.random.normal(ks[5], (D, D), dtype) * 0.02
    bk = jax.random.normal(ks[6], (D,), dtype) * 0.02
    wv = jax.random.normal(ks[7], (D, D), dtype) * 0.02
    bv = jax.random.normal(ks[8], (D,), dtype) * 0.02
    wo = jax.random.normal(ks[9], (D, D), dtype) * 0.02
    bo = jax.random.normal(ks[10], (D,), dtype) * 0.02
    wg = jax.random.normal(ks[11], (hd, 1), dtype) * 0.02
    bg = jax.random.normal(ks[12], (1,), dtype) * 0.02
    return (query, key, inputs, wq, bq, wk, bk, wv, bv, wo, bo, wg, bg)


def workload(query, key, inputs, wq, bq, wk, bk, wv, bv, wo, bo, wg, bg):
    B, Tq, _ = query.shape
    D = wq.shape[1]
    hd = wg.shape[0]
    N = D // hd  # head count implied by the gate's input width

    q = jnp.matmul(query, wq, precision=_HI) + bq
    k = jnp.matmul(key, wk, precision=_HI) + bk
    v = jnp.matmul(inputs, wv, precision=_HI) + bv

    def split(x):
        return x.reshape(B, -1, N, hd).transpose(0, 2, 1, 3)  # (B, N, T, hd)

    qh, kh, vh = split(q), split(k), split(v)
    # Attention(value=None) returns unscaled, un-normalized dot products.
    probs = jnp.einsum('bnqd,bnkd->bnqk', qh, kh, precision=_HI)
    mix = jnp.einsum('bnqk,bnkd->bnqd', probs, vh, precision=_HI)
    mix = mix.transpose(0, 2, 1, 3)  # (B, Tq, N, hd)
    gate = jnp.matmul(mix, wg, precision=_HI) + bg  # (B, Tq, N, 1)
    # torch F.softmax with implicit dim on a 4-D tensor uses dim=1 (Tq).
    gate = jax.nn.softmax(gate, axis=1)
    mixed = jnp.sum(probs.transpose(0, 2, 1, 3) * gate, axis=-2)  # (B, Tq, Tk)
    out = jnp.matmul(mixed, inputs, precision=_HI)  # raw inputs, not wv(inputs)
    return jnp.matmul(out, wo, precision=_HI) + bo
