"""Dynamic per-token int8 absmax activation quant -> int8 x int8 matmul (int32 acc) -> per-token * per-channel dequant."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w8a8_dynamic_per_token_matmul',
    'M': 8192,        # tokens
    'K': 4096,        # hidden size (contraction)
    'N': 14336,       # output channels
    'qmax': 127.0,    # symmetric int8 range
    'eps': 1e-8,      # floor on per-token absmax
    'w_scale_std': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    qmax = int(CONFIG['qmax'])
    k_x, k_w, k_s = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k_x, (M, K), dtype)
    w_q = jax.random.randint(k_w, (N, K), -qmax, qmax + 1, dtype=jnp.int32).astype(jnp.int8)
    w_scale = (jnp.abs(jax.random.normal(k_s, (N,), jnp.float32)) * CONFIG['w_scale_std']
               + 1e-3 * CONFIG['w_scale_std']) / CONFIG['qmax']
    return x, w_q, w_scale


def workload(x, w_q, w_scale):
    qmax = CONFIG['qmax']
    xf = x.astype(jnp.float32)
    absmax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)            # [M,1]
    x_scale = jnp.maximum(absmax, CONFIG['eps']) / qmax               # [M,1]
    x_q = jnp.clip(jnp.round(xf / x_scale), -qmax, qmax).astype(jnp.int8)
    acc = lax.dot_general(
        x_q, w_q,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )                                                                  # [M,N] int32
    out = acc.astype(jnp.float32) * x_scale * w_scale[None, :].astype(jnp.float32)
    return out.astype(x.dtype)
