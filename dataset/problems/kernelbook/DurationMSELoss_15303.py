"""DurationMSELoss: mean((outputs - log(targets + offset))^2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DurationMSELoss_15303',
    'shape': (16, 128, 128, 128),
    'offset': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    outputs = jax.random.uniform(k1, CONFIG['shape'], dtype)
    targets = jax.random.uniform(k2, CONFIG['shape'], dtype)
    return outputs, targets


def workload(outputs, targets):
    log_targets = jnp.log(targets.astype(outputs.dtype) + CONFIG['offset'])
    return jnp.mean(jnp.square(outputs - log_targets))
