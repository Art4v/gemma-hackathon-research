"""AvgPool2d -> flatten -> Linear -> log_softmax classification head (NCHW input)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PredictionHead_7510',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pool_kernel': 256,  # AvgPool2d kernel and stride; equals input_shape[2] (height)
    'num_classes': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['channels'], CONFIG['num_classes']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['num_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    k = x.shape[2]  # kernel = stride = height, as in nn.AvgPool2d(input_shape[2])
    pooled = lax.reduce_window(
        x,
        0.0,
        lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, k, k),
        padding='VALID',
    ) / (k * k)
    flat = pooled.reshape(pooled.shape[0], -1)
    logits = flat @ w + b
    return jax.nn.log_softmax(logits, axis=1)
