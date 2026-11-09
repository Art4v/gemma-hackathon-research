"""[bf16 variant] GaussianSmearing: tile(x, num_freqs) - offset -> square -> scale by coeff -> exp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GaussianSmearing_1703',
    'batch': 8192,
    'in_features': 2048,
    'num_freqs': 50,
    'start': 0.0,
    'end': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['in_features']), dtype=dtype)
    centers = jnp.linspace(CONFIG['start'], CONFIG['end'], CONFIG['num_freqs'], dtype=dtype)
    # offset[f * F + i] = centers[f], matching offset.view(-1,1).repeat(1,F).view(1,-1)
    offset = jnp.repeat(centers, CONFIG['in_features']).reshape(1, -1)
    return x, offset


def workload(x, offset):
    n = CONFIG['num_freqs']
    step = (CONFIG['end'] - CONFIG['start']) / (n - 1)
    coeff = -0.5 / step ** 2
    d = jnp.tile(x, (1, n)) - offset
    return jnp.exp(coeff * d * d)
