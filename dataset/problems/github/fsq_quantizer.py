"""[bf16 variant] Finite Scalar Quantization: Linear -> shifted-tanh bound -> round -> normalize -> mixed-radix indices -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fsq_quantizer',
    'batch': 16,
    'seq_len': 1024,
    'dim': 1024,
    'levels': (8, 8, 8, 5, 5, 5),
    'bound_eps': 1e-3,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    b, s, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    n = len(CONFIG['levels'])
    x = jax.random.normal(k[0], (b, s, d), dtype)
    w_in = jax.random.normal(k[1], (d, n), dtype) * 0.02
    b_in = jax.random.normal(k[2], (n,), dtype) * 0.02
    w_out = jax.random.normal(k[3], (n, d), dtype) * 0.02
    b_out = jax.random.normal(k[4], (d,), dtype) * 0.02
    return x, w_in, b_in, w_out, b_out


def workload(x, w_in, b_in, w_out, b_out):
    levels_i = jnp.array(CONFIG['levels'], dtype=jnp.int32)
    basis = jnp.cumprod(jnp.array((1,) + tuple(CONFIG['levels'][:-1]), dtype=jnp.int32)).astype(jnp.float32)
    levels = levels_i.astype(jnp.float32)
    half_width = (levels_i // 2).astype(jnp.float32)

    z = jnp.matmul(x, w_in, precision=lax.Precision.HIGHEST) + b_in

    half_l = (levels - 1.0) * (1.0 + CONFIG['bound_eps']) / 2.0
    offset = jnp.where(levels_i % 2 == 0, 0.5, 0.0).astype(jnp.float32)
    shift = jnp.arctanh(offset / half_l)
    bounded = jnp.tanh(z + shift) * half_l - offset
    codes = jnp.round(bounded) / half_width

    indices = jnp.round(jnp.sum((codes * half_width + half_width) * basis, axis=-1)).astype(jnp.int32)
    out = jnp.matmul(codes, w_out, precision=lax.Precision.HIGHEST) + b_out
    return out, indices
