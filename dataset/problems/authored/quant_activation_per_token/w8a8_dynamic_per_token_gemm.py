"""Dynamic per-token int8 activation quantization followed by int8xint8 GEMM with per-channel weight dequantization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w8a8_dynamic_per_token_gemm',
    'M': 8192,
    'K': 8192,
    'N': 28672,
    'qmax': 127.0,
    'qmin': -128.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w_q = jax.random.randint(k2, (K, N), minval=-127, maxval=128, dtype=jnp.int8)
    w_scale = jax.random.uniform(k3, (N,), dtype=jnp.float32, minval=0.001, maxval=0.02)
    return x, w_q, w_scale

def workload(x, w_q, w_scale):
    qmax = CONFIG['qmax']
    qmin = CONFIG['qmin']
    eps = CONFIG['eps']

    x_f32 = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
    x_scale = jnp.maximum(absmax, eps) / qmax

    x_q = jnp.clip(jnp.round(x_f32 / x_scale), qmin, qmax).astype(jnp.int8)

    acc = lax.dot(x_q, w_q, preferred_element_type=jnp.int32)
    out = acc.astype(jnp.float32) * x_scale * w_scale[None, :]
    return out.astype(x.dtype)
