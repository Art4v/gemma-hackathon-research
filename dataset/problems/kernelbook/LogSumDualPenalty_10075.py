"""LogSumDualPenalty: sqrt(eps^2 + 4*eta) -> 2 * sum(log((sqrt + eps) / 2) - (sqrt - eps)^2 / (4 * eta))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogSumDualPenalty_10075',
    'shape': (16, 128, 128, 128),
    'epsilon': 1.0,
    'eta_min': 1e-3,
    'eta_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    eta = jax.random.uniform(
        key,
        CONFIG['shape'],
        dtype,
        minval=CONFIG['eta_min'],
        maxval=CONFIG['eta_max'],
    )
    return (eta,)


def workload(eta):
    epsilon = CONFIG['epsilon']
    root = jnp.sqrt(epsilon ** 2 + 4 * eta)
    return 2 * jnp.sum(jnp.log((root + epsilon) / 2) - (root - epsilon) ** 2 / (4 * eta))
