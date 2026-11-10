"""[bf16 variant] BCE focal loss: sigmoid -> alpha/gamma-weighted binary cross-entropy -> sum -> scale by loss_weight / 54."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEFocalLoss_16817',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'gamma': 2,
    'alpha': 0.25,
    'reduction': 'sum',
    'loss_weight': 1.0,
    'divisor': 54.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_x, k_t = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k_x, shape, dtype)
    target = jax.random.uniform(k_t, shape, dtype)
    return x, target


def workload(x, target):
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    pt = jax.nn.sigmoid(x)
    loss = (-alpha * (1 - pt) ** gamma * target * jnp.log(pt)
            - (1 - alpha) * pt ** gamma * (1 - target) * jnp.log(1 - pt))
    if CONFIG['reduction'] == 'elementwise_mean':
        loss = jnp.mean(loss)
    elif CONFIG['reduction'] == 'sum':
        loss = jnp.sum(loss)
    return loss * CONFIG['loss_weight'] / CONFIG['divisor']
