"""[bf16 variant] GeometricMean: exp(mean(log(relu(x)), axis=dim))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GeometricMean_1127',
    'shape': (8, 32, 32, 64, 128),
    'dim': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    log_x = jnp.log(jax.nn.relu(x))
    return jnp.exp(jnp.mean(log_x, axis=CONFIG['dim']))
