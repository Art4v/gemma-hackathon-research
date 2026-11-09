"""MeanStd_26: view(B, C, -1) -> mean over spatial, E[x^2] - mean^2 -> concat([mean, var], dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MeanStd_26',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    b, c = x.shape[0], x.shape[1]
    x = x.reshape(b, c, -1)
    mean_x = jnp.mean(x, axis=2)
    # One-pass biased variance, kept as in the torch module (not jnp.var) to match its numerics.
    var_x = jnp.mean(x ** 2, axis=2) - mean_x * mean_x
    return jnp.concatenate([mean_x, var_x], axis=1)
