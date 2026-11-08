"""Dynamic per-token absmax int8 activation quantization -> int8 x int8 matmul (int32 acc) -> per-token * per-channel dequantization."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w8a8_per_token_dynamic_quant_matmul',
    'M': 8192,          # tokens (batch * seq)
    'K': 4096,          # hidden size
    'N': 14336,         # output features (Llama-3-8B intermediate)
    'qmax': 127,        # symmetric int8 range [-127, 127]
    'eps': 1e-8,        # floor on absmax to avoid division by zero
    'w_scale_base': 0.02 / 127.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, ks = jax.random.split(key, 3)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    qmax = CONFIG['qmax']
    x = jax.random.normal(kx, (M, K), dtype=jnp.float32).astype(dtype)
    w_q = jax.random.randint(kw, (N, K), -qmax, qmax + 1, dtype=jnp.int32).astype(jnp.int8)
    w_scale = jax.random.uniform(ks, (N,), dtype=jnp.float32, minval=0.5, maxval=1.5) * CONFIG['w_scale_base']
    return x, w_q, w_scale


def workload(x, w_q, w_scale):
    qmax = float(CONFIG['qmax'])
    eps = CONFIG['eps']
    xf = x.astype(jnp.float32)
    # Dynamic per-token (row-wise) symmetric absmax quantization.
    absmax = jnp.max(jnp.abs(xf), axis=-1, keepdims=True)
    x_scale = jnp.maximum(absmax, eps) / qmax                      # [M, 1]
    x_q = jnp.clip(jnp.round(xf / x_scale), -qmax, qmax).astype(jnp.int8)
    # int8 x int8 -> int32 matmul: contract K of x_q [M,K] with K of w_q [N,K].
    acc = lax.dot_general(
        x_q, w_q,
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.int32,
    )                                                               # [M, N] int32
    # Dequantize with per-token and per-output-channel scales.
    out = acc.astype(jnp.float32) * x_scale * w_scale.astype(jnp.float32)[None, :]
    return out.astype(x.dtype), x_scale[:, 0]
