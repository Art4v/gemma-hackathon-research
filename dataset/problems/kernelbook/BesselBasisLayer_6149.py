"""BesselBasisLayer: scale by cutoff -> polynomial envelope (masked at d>=1) * sin(freq * d)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BesselBasisLayer_6149',
    'batch': 16,
    'd1': 128,
    'd2': 128,
    'd3': 128,
    'num_radial': 4,
    'cutoff': 4.0,
    'envelope_exponent': 5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3']), dtype)
    frequencies = (jnp.pi * jnp.arange(1, CONFIG['num_radial'] + 1)).astype(dtype)
    return x, frequencies


def _envelope(x):
    p = CONFIG['envelope_exponent'] + 1
    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2
    env_val = 1 / x + a * x ** (p - 1) + b * x ** p + c * x ** (p + 1)
    return jnp.where(x < 1, env_val, jnp.zeros_like(x))


def workload(x, frequencies):
    d_scaled = (x / CONFIG['cutoff'])[..., None]
    return _envelope(d_scaled) * jnp.sin(frequencies * d_scaled)
