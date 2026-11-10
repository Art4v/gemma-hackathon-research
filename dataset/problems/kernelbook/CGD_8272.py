"""CGD channel attention: global avg/max pool, softmax, matmul, tanh gating chain, residual rescale of NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CGD_8272',
    'batch': 8,
    'in_channels': 128,
    'height': 256,
    'width': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w0, k_w1, k_w2 = jax.random.split(key, 4)
    b = CONFIG['batch']
    c = CONFIG['in_channels']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (b, c, CONFIG['height'], CONFIG['width']), dtype)
    w0 = jax.random.normal(k_w0, (c, 1), dtype) * scale
    w1 = jax.random.normal(k_w1, (c, 1), dtype) * scale
    w2 = jax.random.normal(k_w2, (c, 1), dtype) * scale
    bias0 = jnp.zeros((1, c, 1, 1), dtype)
    bias1 = jnp.zeros((1, c, 1, 1), dtype)
    bias2 = jnp.zeros((1, c, 1, 1), dtype)
    return x, w0, w1, w2, bias0, bias1, bias2


def workload(x, w0, w1, w2, bias0, bias1, bias2):
    b, c = x.shape[0], x.shape[1]
    x0 = jnp.mean(x, axis=(2, 3), keepdims=True)
    x1 = jnp.max(x, axis=(2, 3), keepdims=True)
    x0_s = jax.nn.softmax(x0, axis=1)
    y0 = jnp.matmul(x0.reshape(b, c), w0).reshape(b, 1, 1, 1)
    y1 = jnp.matmul(x1.reshape(b, c), w1).reshape(b, 1, 1, 1)
    y0_s = jnp.tanh(y0 * x0_s + bias0)
    y1_s = jnp.tanh(y1 * x0_s + bias1)
    y2 = jnp.matmul(y1_s.reshape(b, c), w2).reshape(b, 1, 1, 1)
    y2_s = jnp.tanh(y2 * y0_s + bias2).reshape(b, c, 1, 1)
    return x * (y2_s + 1)
