"""Non-causal linear attention with elu(x) + 1 feature map and denominator normalization."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linear_attention_elu_noncausal',
    'batch': 8,
    'seq_len': 8192,
    'num_heads': 16,
    'head_dim': 64,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v = jax.random.split(key, 3)
    shape = (CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim'])
    q = jax.random.normal(k_q, shape, dtype=dtype)
    k = jax.random.normal(k_k, shape, dtype=dtype)
    v = jax.random.normal(k_v, shape, dtype=dtype)
    return q, k, v


def workload(q, k, v):
    orig_dtype = q.dtype
    eps = CONFIG['eps']

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    phi_q = jax.nn.elu(q_f32) + 1.0
    phi_k = jax.nn.elu(k_f32) + 1.0

    kv = jnp.einsum('bshd,bshm->bhdm', phi_k, v_f32)
    num = jnp.einsum('bshd,bhdm->bshm', phi_q, kv)

    k_sum = jnp.sum(phi_k, axis=1)
    den = jnp.einsum('bshd,bhd->bsh', phi_q, k_sum)[..., None]

    out = num / (den + eps)
    return out.astype(orig_dtype)
