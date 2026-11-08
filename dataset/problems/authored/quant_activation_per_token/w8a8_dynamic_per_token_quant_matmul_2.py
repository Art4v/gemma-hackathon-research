"""Dynamic per-token int8 activation quantization followed by int8xint8 matmul and per-token/per-channel dequantization."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w8a8_dynamic_per_token_quant_matmul',
    'batch': 4,
    'seq_len': 2048,
    'in_features': 8192,
    'out_features': 8192,
    'qmax': 127.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    K = CONFIG['in_features']
    N = CONFIG['out_features']
    qmax = int(CONFIG['qmax'])

    x = jax.random.normal(k1, (B, S, K), dtype=dtype)
    w_q = jax.random.randint(k2, (K, N), minval=-qmax, maxval=qmax + 1, dtype=jnp.int8)
    w_scale = jax.random.uniform(k3, (N,), dtype=jnp.float32, minval=1e-4, maxval=1e-2)
    bias = jax.random.normal(k4, (N,), dtype=dtype) * 0.02
    return x, w_q, w_scale, bias

def workload(x, w_q, w_scale, bias):
    qmax = CONFIG['qmax']
    eps = CONFIG['eps']
    orig_dtype = x.dtype

    x_f32 = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(x_f32), axis=-1, keepdims=True)
    x_scale = jnp.maximum(absmax, eps) / qmax

    x_q = jnp.clip(jnp.round(x_f32 / x_scale), -qmax, qmax).astype(jnp.int8)

    acc = lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((2,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )

    out_f32 = acc.astype(jnp.float32) * x_scale * w_scale[None, None, :] + bias.astype(jnp.float32)
    return out_f32.astype(orig_dtype)
