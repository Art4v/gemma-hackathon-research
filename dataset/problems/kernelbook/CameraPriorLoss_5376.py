"""[bf16 variant] CameraPriorLoss: select cameras[:, 0], exp(-x * scale), square, scale by loss_weight, reduce."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CameraPriorLoss_5376',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'camera_index': 0,
    'scale': 10.0,
    'loss_weight': 1.0,
    'reduction': 'mean',
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    cameras = jax.random.uniform(key, shape, dtype)
    return (cameras,)


def workload(cameras):
    selected = cameras[:, CONFIG['camera_index']]
    loss = jnp.exp(-selected * CONFIG['scale'])
    loss = jnp.power(loss, 2) * CONFIG['loss_weight']
    if CONFIG['reduction'] == 'mean':
        loss = jnp.mean(loss)
    elif CONFIG['reduction'] == 'sum':
        loss = jnp.sum(loss)
    return loss
