"""[bf16 variant] HardWeightedSum: ReLU(weights), sum plus eps denominator, broadcast weights * x / denom, sum over the op axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HardWeightedSum_17893',
    'x_shape': (16, 128, 128, 128),
    'weights_shape': (2, 1, 1, 1, 1),
    'eps': 1e-4,
    'sum_axis': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, CONFIG['x_shape'], dtype)
    weights = jnp.ones(CONFIG['weights_shape'], dtype)
    return x, weights


def workload(x, weights):
    weights_num = jnp.maximum(weights, 0.0)
    weights_denom = jnp.sum(weights_num) + CONFIG['eps']
    return jnp.sum(weights_num * x / weights_denom, axis=CONFIG['sum_axis'])
