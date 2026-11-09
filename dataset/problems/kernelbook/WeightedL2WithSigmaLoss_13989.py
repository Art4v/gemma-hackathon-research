"""WeightedL2WithSigmaLoss (default path): nan-masked target, elementwise 0.5 * (input - target)^2."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedL2WithSigmaLoss_13989',
    'd0': 16,
    'd1': 128,
    'd2': 128,
    'd3': 256,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    k_in, k_tgt = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_in, shape, dtype)
    target = jax.random.uniform(k_tgt, shape, dtype)
    return x, target


def workload(x, target):
    # ponytail: code_weights, sigma and weights are all None in this module's default call.
    target = jnp.where(jnp.isnan(target), x, target)
    diff = x - target
    return 0.5 * diff ** 2
