"""[bf16 variant] Walsh-Hadamard rotation (pad to pow2, Sylvester H, 1/sqrt(n) scale, truncate) followed by per-token symmetric int4 fake-quant."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'hadamard_rotation_int4_quant',
    'tokens': 4096,
    'dim': 4096,
    'bits': 4,
    'scale_clamp_min': 1e-5,
    'outlier_stride': 97,
    'outlier_scale': 50.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.normal(key, (CONFIG['tokens'], CONFIG['dim']), dtype)
    x = x.at[:, ::CONFIG['outlier_stride']].multiply(CONFIG['outlier_scale'])
    return (x,)


def _sylvester_hadamard(n, dtype):
    h = jnp.ones((1, 1), dtype)
    while h.shape[0] < n:
        h = jnp.concatenate(
            [jnp.concatenate([h, h], 1), jnp.concatenate([h, -h], 1)], 0)
    return h


def _hadamard_transform(x, scale):
    lead_shape = x.shape[:-1]
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)
    dim_padded = 1 << (dim - 1).bit_length()
    x2 = jnp.pad(x2, ((0, 0), (0, dim_padded - dim)))
    h = _sylvester_hadamard(dim_padded, x2.dtype)
    out = jnp.matmul(x2, h.T, precision=lax.Precision.HIGHEST) * scale
    return out[:, :dim].reshape(*lead_shape, dim)


def workload(x):
    qmax = 2 ** (CONFIG['bits'] - 1) - 1
    y = _hadamard_transform(x, x.shape[-1] ** -0.5)
    s = jnp.maximum(jnp.max(jnp.abs(y), axis=-1, keepdims=True),
                    CONFIG['scale_clamp_min']) / qmax
    return jnp.clip(jnp.round(y / s), -qmax - 1, qmax) * s
