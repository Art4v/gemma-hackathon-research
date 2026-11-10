"""HardSigmoid: nested where of x >= 3 -> 1, x <= -3 -> 0, else slope * x + offset."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HardSigmoid_8002',
    'shape': (32, 128, 128, 128),
    'minval': -4.0,
    'maxval': 4.0,
    'slope': 0.1666667,
    'offset': 0.5,
    'upper_threshold': 3.0,
    'lower_threshold': -3.0,
    'high_value': 1.0,
    'low_value': 0.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        CONFIG['shape'],
        dtype,
        minval=CONFIG['minval'],
        maxval=CONFIG['maxval'],
    )
    return (x,)


def workload(x):
    high = jnp.asarray(CONFIG['high_value'], x.dtype)
    low = jnp.asarray(CONFIG['low_value'], x.dtype)
    slope = jnp.asarray(CONFIG['slope'], x.dtype)
    offset = jnp.asarray(CONFIG['offset'], x.dtype)
    upper = jnp.asarray(CONFIG['upper_threshold'], x.dtype)
    lower = jnp.asarray(CONFIG['lower_threshold'], x.dtype)
    return jnp.where(
        x >= upper,
        high,
        jnp.where(x <= lower, low, x * slope + offset),
    )
