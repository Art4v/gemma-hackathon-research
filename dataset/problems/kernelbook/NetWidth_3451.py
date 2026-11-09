"""Conv2d+tanh+maxpool2 -> Conv2d+tanh+maxpool2 -> view(-1,1024) -> Linear+tanh -> Linear (NetWidth)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NetWidth_3451',
    'batch': 2048,
    'in_channels': 3,
    'height': 64,
    'width': 64,
    'conv1_out': 32,
    'conv2_out': 16,
    'kernel_size': 3,
    'padding': 1,
    'pool': 2,
    'fc_in': 16 * 8 * 8,
    'fc1_out': 32,
    'fc2_out': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 9)
    k = c['kernel_size']
    x = jax.random.uniform(ks[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w_c1 = jax.random.normal(ks[1], (c['conv1_out'], c['in_channels'], k, k), dtype) * 0.02
    b_c1 = jax.random.normal(ks[2], (c['conv1_out'],), dtype) * 0.02
    w_c2 = jax.random.normal(ks[3], (c['conv2_out'], c['conv1_out'], k, k), dtype) * 0.02
    b_c2 = jax.random.normal(ks[4], (c['conv2_out'],), dtype) * 0.02
    w_f1 = jax.random.normal(ks[5], (c['fc_in'], c['fc1_out']), dtype) * 0.02
    b_f1 = jax.random.normal(ks[6], (c['fc1_out'],), dtype) * 0.02
    w_f2 = jax.random.normal(ks[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    b_f2 = jax.random.normal(ks[8], (c['fc2_out'],), dtype) * 0.02
    return x, w_c1, b_c1, w_c2, b_c2, w_f1, b_f1, w_f2, b_f2


def _conv_tanh_pool(h, w, b):
    p = CONFIG['padding']
    s = CONFIG['pool']
    h = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    h = jnp.tanh(h + b)
    return lax.reduce_window(h, -jnp.inf, lax.max, (1, s, s, 1), (1, s, s, 1), 'VALID')


def workload(x, w_c1, b_c1, w_c2, b_c2, w_f1, b_f1, w_f2, b_f2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_tanh_pool(h, w_c1, b_c1)
    h = _conv_tanh_pool(h, w_c2, b_c2)
    # PyTorch view on NCHW: with 64x64 input each sample yields 4 rows of fc_in.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, w_f1.shape[0])
    h = jnp.tanh(h @ w_f1 + b_f1)
    return h @ w_f2 + b_f2
