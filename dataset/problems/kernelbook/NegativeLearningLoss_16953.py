"""NegativeLearningLoss: mask(p < thr) * -log(1 - p + eps), summed and divided by the mask count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NegativeLearningLoss_16953',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'threshold': 0.05,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    predict = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (predict,)


def workload(predict):
    mask = (predict < CONFIG['threshold']).astype(predict.dtype)
    loss_item = -1.0 * mask * jnp.log(1.0 - predict + CONFIG['eps'])
    return jnp.sum(loss_item) / jnp.sum(mask)
