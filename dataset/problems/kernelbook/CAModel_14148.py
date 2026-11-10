"""[bf16 variant] CAModel: Conv2d 1x1 (12->232) -> ReLU -> Conv2d 1x1 (232->4)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CAModel_14148',
    'batch': 8,
    'env_d': 4,
    'in_channels': 12,
    'hidden_channels': 232,
    'out_channels': 4,
    'height': 512,
    'width': 512,
    'kernel_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.key(0)
    kx, kw1, kb1, kw2, kb2 = jax.random.split(k, 5)
    ks = CONFIG['kernel_size']
    cin, ch, cout = CONFIG['in_channels'], CONFIG['hidden_channels'], CONFIG['out_channels']
    x = jax.random.uniform(kx, (CONFIG['batch'], cin, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(kw1, (ks, ks, cin, ch), dtype) * 0.02
    b1 = jax.random.normal(kb1, (ch,), dtype) * 0.02
    w2 = jax.random.normal(kw2, (ks, ks, ch, cout), dtype) * 0.02
    b2 = jax.random.normal(kb2, (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1))
    h = _conv(h, w2, b2)
    return jnp.transpose(h, (0, 3, 1, 2))
