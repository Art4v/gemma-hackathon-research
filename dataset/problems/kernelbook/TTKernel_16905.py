"""Bilinear (no bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TTKernel_16905',
    'lead_shape': (8, 64, 128),
    'r_i': 128,
    'm': 128,
    'r_j': 128,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    lead = CONFIG['lead_shape']
    x1 = jax.random.uniform(k1, (*lead, CONFIG['r_i']), dtype)
    x2 = jax.random.uniform(k2, (*lead, CONFIG['m']), dtype)
    w = jax.random.normal(k3, (CONFIG['r_j'], CONFIG['r_i'], CONFIG['m']), dtype) * 0.02
    return x1, x2, w


def workload(x1, x2, w):
    # w is in PyTorch nn.Bilinear layout (out, in1, in2).
    y = jnp.einsum('...i,kij,...j->...k', x1, w, x2, precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y)
