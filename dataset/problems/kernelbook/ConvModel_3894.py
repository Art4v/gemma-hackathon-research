"""Conv2d+ReLU -> Conv2d+ReLU -> flatten -> Linear+ReLU -> Linear+ReLU -> Linear -> softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvModel_3894',
    'batch': 256,
    'height': 126,
    'width': 126,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 2,
    'padding': 1,
    'fc1_out': 120,
    'fc2_out': 84,
    'output_dim': 128,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    p = c['padding']
    h2 = c['height'] + 2 * (2 * p - k + 1)
    w2 = c['width'] + 2 * (2 * p - k + 1)
    fc1_in = c['conv2_out'] * h2 * w2
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['height'], c['width']), dtype)
    conv1_w = jax.random.normal(keys[1], (c['conv1_out'], 1, k, k), dtype) * 0.02
    conv1_b = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    conv2_w = jax.random.normal(keys[3], (c['conv2_out'], c['conv1_out'], k, k), dtype) * 0.02
    conv2_b = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    fc1_w = jax.random.normal(keys[5], (fc1_in, c['fc1_out']), dtype) * 0.02
    fc1_b = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fc2_w = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fc2_b = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fc3_w = jax.random.normal(keys[9], (c['fc2_out'], c['output_dim']), dtype) * 0.02
    fc3_b = jax.random.normal(keys[10], (c['output_dim'],), dtype) * 0.02
    return (x, conv1_w, conv1_b, conv2_w, conv2_b,
            fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)


def _conv_relu(x_nhwc, w_oihw, b):
    p = CONFIG['padding']
    w_hwio = jnp.transpose(w_oihw, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(x, conv1_w, conv1_b, conv2_w, conv2_b,
             fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    h = x[..., None]  # unsqueeze(1) in NCHW == trailing channel in NHWC
    h = _conv_relu(h, conv1_w, conv1_b)
    h = _conv_relu(h, conv2_w, conv2_b)
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(h.shape[0], -1)  # torch flatten order (C,H,W)
    h = jax.nn.relu(h @ fc1_w + fc1_b)
    h = jax.nn.relu(h @ fc2_w + fc2_b)
    return jax.nn.softmax(h @ fc3_w + fc3_b, axis=1)
