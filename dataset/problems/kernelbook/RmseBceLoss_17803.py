"""[bf16 variant] RmseBceLoss: 2 * sqrt(MSE(sigmoid(pred[0]), label[0]) + smooth) + BCEWithLogits(pred[1], label[1])."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RmseBceLoss_17803',
    'pair': 2,
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'smooth': 1e-6,
    'rmse_weight': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['pair'], CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k1, k2 = jax.random.split(jax.random.key(0))
    prediction = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return prediction, label


def workload(prediction, label):
    rmse = jnp.sqrt(jnp.mean((jax.nn.sigmoid(prediction[0]) - label[0]) ** 2) + CONFIG['smooth'])
    x, y = prediction[1], label[1]
    # PyTorch's numerically stable BCEWithLogits form.
    bce = jnp.mean(jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x))))
    return CONFIG['rmse_weight'] * rmse + bce
