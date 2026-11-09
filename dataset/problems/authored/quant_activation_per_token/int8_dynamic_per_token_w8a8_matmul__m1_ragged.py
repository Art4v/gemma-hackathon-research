"""[m1 M3_ragged variant of authored/quant_activation_per_token/int8_dynamic_per_token_w8a8_matmul] Same per-token int8 quantize -> int8 matmul -> dequantize, with the token count 8192 -> 8100 (not a multiple of 8 or 128)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'int8_dynamic_per_token_w8a8_matmul_m1_ragged',
    'num_tokens': 8100,
    'hidden_in': 8192,
    'hidden_out': 8192,
    'qmax': 127,
    'eps': 1e-8,
    'w_scale_base': 0.02 / 127.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, ks = jax.random.split(key, 3)
    M, K, N = CONFIG['num_tokens'], CONFIG['hidden_in'], CONFIG['hidden_out']
    q = CONFIG['qmax']
    x = jax.random.normal(kx, (M, K), dtype)
    w_q = jax.random.randint(kw, (K, N), -q, q + 1, dtype=jnp.int8)
    w_scale = jax.random.uniform(ks, (N,), jnp.float32, 0.5, 1.5) * CONFIG['w_scale_base']
    return x, w_q, w_scale


def workload(x, w_q, w_scale):
    qmax = float(CONFIG['qmax'])
    xf = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    x_scale = jnp.maximum(absmax, CONFIG['eps']) / qmax
    x_q = jnp.clip(jnp.round(xf / x_scale), -qmax, qmax).astype(jnp.int8)
    acc = lax.dot_general(
        x_q, w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.int32,
    )
    out = acc.astype(jnp.float32) * x_scale * w_scale.astype(jnp.float32)[None, :]
    return out.astype(x.dtype)
