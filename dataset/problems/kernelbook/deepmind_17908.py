"""[bf16 variant] Conv2d(8,s4)+ReLU -> Conv2d(4,s2)+ReLU -> Conv2d(3,s1)+ReLU -> view(-1, 32*7*7) (DeepMind Atari trunk)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'deepmind_17908',
    'batch': 256,
    'in_channels': 4,
    'height': 144,
    'width': 144,
    'c1': 32, 'k1': 8, 's1': 4,
    'c2': 64, 'k2': 4, 's2': 2,
    'c3': 32, 'k3': 3, 's3': 1,
    'flat_dim': 32 * 7 * 7,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k1, (c['c1'], c['in_channels'], c['k1'], c['k1']), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['c2'], c['c1'], c['k2'], c['k2']), dtype) * 0.02
    w3 = jax.random.normal(k3, (c['c3'], c['c2'], c['k3'], c['k3']), dtype) * 0.02
    b1 = jnp.zeros((c['c1'],), dtype)
    b2 = jnp.zeros((c['c2'],), dtype)
    b3 = jnp.zeros((c['c3'],), dtype)
    return x, w1, b1, w2, b2, w3, b3


def _conv_relu(x, w_oihw, b, stride):
    w = jnp.transpose(w_oihw, (2, 3, 1, 0))  # OIHW -> HWIO
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2, w3, b3):
    c = CONFIG
    h = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    h = _conv_relu(h, w1, b1, c['s1'])
    h = _conv_relu(h, w2, b2, c['s2'])
    h = _conv_relu(h, w3, b3, c['s3'])
    h = jnp.transpose(h, (0, 3, 1, 2))  # back to NCHW so the flatten matches torch .view
    return h.reshape(-1, c['flat_dim'])
