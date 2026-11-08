"""Causal depthwise short conv1d (K=4) + bias -> SiLU -> multiply by SiLU(z) gate (Mamba/Hyena short-conv gate)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'short_conv_silu_gate',
    'batch': 4,
    'seq_len': 4096,
    'd_inner': 5120,
    'kernel_size': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    K = CONFIG['kernel_size']
    key = jax.random.key(0)
    kx, kz, kw, kb = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, L, D), dtype)
    z = jax.random.normal(kz, (B, L, D), dtype)
    w = jax.random.normal(kw, (K, D), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(kb, (D,), dtype) * CONFIG['weight_scale']
    return x, z, w, b


def workload(x, z, w, b):
    K = w.shape[0]
    L = x.shape[1]
    xf = x.astype(jnp.float32)
    wf = w.astype(jnp.float32)
    xpad = jnp.pad(xf, ((0, 0), (K - 1, 0), (0, 0)))
    acc = jnp.zeros(xf.shape, jnp.float32)
    for k in range(K):
        acc = acc + xpad[:, k:k + L, :] * wf[k][None, None, :]
    acc = acc + b.astype(jnp.float32)[None, None, :]
    u = jax.nn.silu(acc)
    g = jax.nn.silu(z.astype(jnp.float32))
    return (u * g).astype(x.dtype)
