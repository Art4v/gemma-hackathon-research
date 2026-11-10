"""[bf16 variant] Laplacian-pyramid block: bilinear 0.5x/2x resize, two residual Conv2d-InstanceNorm-ReLU branches, bilinear 2x merge."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'lrBLock_l2_8758',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'window_size': 3,
    'stride': 1,
    'padding': 1,
    'down_scale': 0.5,
    'up_scale': 2.0,
    'eps': 1e-5,
    'weight_scale': 0.02,
    'num_convs': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    k = CONFIG['window_size']
    n = CONFIG['num_convs']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 2 * n + 1)
    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    params = []
    for i in range(n):
        params.append(jax.random.normal(keys[2 * i + 1], (c, c, k, k), dtype) * scale)
        params.append(jax.random.normal(keys[2 * i + 2], (c,), dtype) * scale)
    return (x, *params)


def _resize_axis(x, axis, scale):
    n = x.shape[axis]
    m = int(n * scale)
    src = (jnp.arange(m, dtype=x.dtype) + 0.5) / scale - 0.5
    src = jnp.maximum(src, 0.0)
    floor = jnp.floor(src)
    lam = src - floor
    i0 = floor.astype(jnp.int32)
    i1 = jnp.minimum(i0 + 1, n - 1)
    shape = [1] * x.ndim
    shape[axis] = m
    lam = lam.reshape(shape)
    return jnp.take(x, i0, axis=axis) * (1.0 - lam) + jnp.take(x, i1, axis=axis) * lam


def _bilinear(x, scale):
    return _resize_axis(_resize_axis(x, 2, scale), 3, scale)


def _conv2d(x, weight, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        weight,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return jnp.transpose(y + bias, (0, 3, 1, 2))


def _instance_norm(x):
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(2, 3), keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps'])


def _res_block(x, w1, b1, w2, b2):
    y = jax.nn.relu(_instance_norm(_conv2d(x, w1, b1)))
    y = _instance_norm(_conv2d(y, w2, b2))
    return jax.nn.relu(y + x)


def workload(x, w_l1_1, b_l1_1, w_l1_2, b_l1_2, w_l2_1, b_l2_1, w_l2_2, b_l2_2):
    x_down2 = _bilinear(x, CONFIG['down_scale'])
    x_reup = _bilinear(x_down2, CONFIG['up_scale'])
    laplace_1 = x - x_reup
    scale1 = _res_block(x_down2, w_l1_1, b_l1_1, w_l1_2, b_l1_2)
    scale2 = _res_block(laplace_1, w_l2_1, b_l2_1, w_l2_2, b_l2_2)
    return _bilinear(scale1, CONFIG['up_scale']) + scale2
