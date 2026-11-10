"""LeakyClamp: leaky_relu(x), then cap - leaky_relu(cap - y) soft upper clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LeakyClamp_3779',
    'shape': (8, 128, 256, 256),
    'cap': 4.0,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    slope = CONFIG['negative_slope']
    cap = jnp.asarray(CONFIG['cap'], x.dtype)
    y = jax.nn.leaky_relu(x, negative_slope=slope)
    return -jax.nn.leaky_relu(-y + cap, negative_slope=slope) + cap
