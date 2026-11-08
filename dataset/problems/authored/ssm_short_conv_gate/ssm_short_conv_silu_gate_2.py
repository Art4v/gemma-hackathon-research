"""Short causal depthwise conv1d (K=4) + bias + SiLU, gated by SiLU of a second branch (Mamba/Hyena style)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ssm_short_conv_silu_gate',
    'batch': 4,
    'seq_len': 4096,
    'd_inner': 5120,
    'kernel_size': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    K = CONFIG['kernel_size']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    kx, kz, kw, kb = jax.random.split(key, 4)
    x = jax.random.normal(kx, (B, L, D), dtype)
    z = jax.random.normal(kz, (B, L, D), dtype)
    conv_w = jax.random.normal(kw, (K, D), dtype) * s
    conv_b = jax.random.normal(kb, (D,), dtype) * s
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
    u = jax.nn.silu(acc)
    g = jax.nn.silu(z.astype(jnp.float32))
    return (u * g).astype(x.dtype)
