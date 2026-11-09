"""[bf16 variant] resBlock: conv3x3 -> InstanceNorm -> ReLU -> conv3x3 -> InstanceNorm -> +residual -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'resBlock_8744',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c, k = CONFIG['channels'], CONFIG['kernel_size']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (k, k, c, c), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c,), dtype) * 0.02
    w2 = jax.random.normal(k2, (k, k, c, c), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(h, w, b):
    pad = w.shape[0] // 2
    y = lax.conv_general_dilated(
        h, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _instance_norm(h):
    mean = jnp.mean(h, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=(1, 2), keepdims=True)
    return (h - mean) * lax.rsqrt(var + CONFIG['eps'])


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    y = jax.nn.relu(_instance_norm(_conv(h, w1, b1)))
    y = _instance_norm(_conv(y, w2, b2))
    out = jax.nn.relu(y + h)
    return jnp.transpose(out, (0, 3, 1, 2))
