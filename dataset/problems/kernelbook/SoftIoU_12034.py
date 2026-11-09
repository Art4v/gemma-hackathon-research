"""SoftIoU loss: sigmoid(pred), masked intersection over masked max-union, per-sample reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftIoU_12034',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'from_sigmoid': False,
    'ignore_label': -1,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    pred = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return pred, label


def workload(pred, label):
    label = label.reshape(pred.shape)
    sample_weight = (label != CONFIG['ignore_label']).astype(pred.dtype)
    if not CONFIG['from_sigmoid']:
        pred = jax.nn.sigmoid(pred)
    inter = jnp.sum(pred * label * sample_weight, axis=(1, 2, 3))
    union = jnp.sum(jnp.maximum(pred, label) * sample_weight, axis=(1, 2, 3))
    return 1.0 - inter / (union + CONFIG['eps'])
