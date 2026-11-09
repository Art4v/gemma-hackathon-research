"""DurationPredictorLoss: MSE(outputs, log(targets + offset)) with mean reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DurationPredictorLoss_3086',
    'shape': (16, 128, 128, 128),
    'offset': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    outputs = jax.random.uniform(k_out, CONFIG['shape'], dtype)
    targets = jax.random.uniform(k_tgt, CONFIG['shape'], dtype)
    return outputs, targets


def workload(outputs, targets):
    log_targets = jnp.log(targets.astype(outputs.dtype) + CONFIG['offset'])
    return jnp.mean(jnp.square(outputs - log_targets))
