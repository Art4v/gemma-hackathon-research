"""Dynamic per-token int8 activation quantization followed by int8xint8 matmul and per-token/per-channel dequantization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w8a8_dynamic_per_token_quant_matmul',
    'M': 4096,
    'K': 8192,
    'N': 28672,
    'qmax': 127.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    qmax = int(CONFIG['qmax'])

    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w_q = jax.random.randint(k2, (K, N), minval=-qmax, maxval=qmax + 1, dtype=jnp.int8)
    w_scale = jax.random.uniform(k3, (N,), dtype=jnp.float32, minval=1e-4, maxval=2e-2) * 0.02
    return x, w_q, w_scale

def workload(x, w_q, w_scale):
    qmax = CONFIG['qmax']
    eps = CONFIG['eps']

    x_f32 = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
    x_scale = jnp.maximum(absmax, eps) / qmax

    x_q = jnp.clip(jnp.round(x_f32 / x_scale), -qmax, qmax).astype(jnp.int8)

    acc = lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )

    out_f32 = acc.astype(jnp.float32) * x_scale * w_scale[None, :]
    return out_f32.astype(x.dtype)
