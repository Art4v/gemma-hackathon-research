"""[bf16 variant] ContrastiveDistanceLoss: (1-t)*p^2 + t*clamp(margin-p,0)^2 -> sum / 2 / batch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveDistanceLoss_16263',
    'batch': 32,
    'channels': 64,
    'height': 128,
    'width': 128,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_pred, k_true = jax.random.split(jax.random.key(0))
    distance_pred = jax.random.uniform(k_pred, shape, dtype)
    distance_true = jax.random.uniform(k_true, shape, dtype)
    return distance_pred, distance_true


def workload(distance_pred, distance_true):
    bs = distance_true.shape[0]
    margin_distance = jnp.maximum(CONFIG['margin'] - distance_pred, 0.0)
    loss = (1.0 - distance_true) * distance_pred ** 2 + distance_true * margin_distance ** 2
    return jnp.sum(loss) / 2.0 / bs
