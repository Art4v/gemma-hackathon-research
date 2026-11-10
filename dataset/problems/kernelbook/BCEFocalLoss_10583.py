"""Binary focal loss: sigmoid -> pow -> log -> weighted elementwise sum -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BCEFocalLoss_10583',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'gamma': 2,
    'alpha': 0.25,
    'reduction': 'elementwise_mean',
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_input, key_target = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_input, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return x, target


def workload(x, target):
    gamma = CONFIG['gamma']
    alpha = CONFIG['alpha']
    reduction = CONFIG['reduction']
    pt = jax.nn.sigmoid(x)
    loss = (
        -alpha * (1 - pt) ** gamma * target * jnp.log(pt)
        - (1 - alpha) * pt ** gamma * (1 - target) * jnp.log(1 - pt)
    )
    if reduction == 'elementwise_mean':
        loss = jnp.mean(loss)
    elif reduction == 'sum':
        loss = jnp.sum(loss)
    return loss
