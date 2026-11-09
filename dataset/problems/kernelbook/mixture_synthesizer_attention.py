"""[bf16 variant] Mixture synthesizer attention: QKV linear, bmm(q, k^T) + learned (L, L) attention, softmax, bmm with v."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mixture_synthesizer_attention',
    'batch': 16,
    'sentence_length': 1024,
    'in_dims': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b, l, d = CONFIG['batch'], CONFIG['sentence_length'], CONFIG['in_dims']
    ks = jax.random.split(jax.random.key(0), 8)
    x = jax.random.uniform(ks[0], (b, l, d), dtype)
    attn = jax.random.normal(ks[1], (1, l, l), dtype) * 0.02
    wq = jax.random.normal(ks[2], (d, d), dtype) * 0.02
    bq = jax.random.normal(ks[3], (d,), dtype) * 0.02
    wk = jax.random.normal(ks[4], (d, d), dtype) * 0.02
    bk = jax.random.normal(ks[5], (d,), dtype) * 0.02
    wv = jax.random.normal(ks[6], (d, d), dtype) * 0.02
    bv = jax.random.normal(ks[7], (d,), dtype) * 0.02
    return x, attn, wq, bq, wk, bk, wv, bv


def workload(x, attn, wq, bq, wk, bk, wv, bv):
    q = x @ wq + bq
    k = x @ wk + bk
    v = x @ wv + bv
    energy = attn + jnp.einsum('bld,bmd->blm', q, k)
    attention = jax.nn.softmax(energy, axis=-1)
    out = jnp.einsum('blm,bmd->bld', attention, v)
    return out, attention
