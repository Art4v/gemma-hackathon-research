"""[bf16 variant] GaussianSmearing: exp(coeff * (dist[..., None] - offset)^2) radial basis expansion."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussianSmearing_15962',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'd3': 128,
    'cutoff_lower': 0.0,
    'cutoff_upper': 5.0,
    'num_rbf': 50,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    dist = jax.random.uniform(key, shape, dtype=dtype)
    offset = jnp.linspace(CONFIG['cutoff_lower'], CONFIG['cutoff_upper'],
                          CONFIG['num_rbf'], dtype=dtype)
    coeff = (-0.5 / (offset[1] - offset[0]) ** 2).astype(dtype)
    return dist, offset, coeff


def workload(dist, offset, coeff):
    d = dist[..., None] - offset
    return jnp.exp(coeff * d * d)
