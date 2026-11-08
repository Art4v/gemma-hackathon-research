"""Causal depthwise conv1d (kernel 4) + bias -> SiLU, gated by SiLU(z) (Mamba/Hyena short-conv gate)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ssm_short_conv_silu_gate',
    'batch': 4,
    'seq_len': 4096,
    'd_inner': 5120,
    'kernel_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    B, L, D, K = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_inner'], CONFIG['kernel_size']
    key = jax.random.key(0)
    kx, kz, kw, kb = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, L, D), dtype)
    z = jax.random.normal(kz, (B, L, D), dtype)
    conv_w = jax.random.normal(kw, (K, D), dtype) * 0.02
    conv_b = jax.random.normal(kb, (D,), dtype) * 0.02
    return x, z, conv_w, conv_b


def workload(x, z, conv_w, conv_b):
    K = conv_w.shape[0]
    L = x.shape[1]
    xf = x.astype(jnp.float32)
    wf = conv_w.astype(jnp.float32)
    xpad = jnp.pad(xf, ((0, 0), (K - 1, 0), (0, 0)))
    acc = jnp.zeros(xf.shape, jnp.float32)
    for k in range(K):
        acc = acc + xpad[:, k:k + L, :] * wf[k][None, None, :]
    acc = acc + conv_b.astype(jnp.float32)[None, None, :]
    out = jax.nn.silu(acc) * jax.nn.silu(z.astype(jnp.float32))
    return out.astype(x.dtype)
