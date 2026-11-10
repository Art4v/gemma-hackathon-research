"""[bf16 variant] Conv2d -> MaxPool -> ReLU -> Conv2d -> MaxPool -> ReLU -> NCHW view(-1, 8) -> Linear -> ReLU -> Linear (CNN_14901)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CNN_14901',
    'batch': 32,
    'in_channels': 1,
    'height': 1024,
    'width': 1024,
    'conv1_channels': 4,
    'conv2_channels': 8,
    'kernel_size': 2,
    'pool_size': 2,
    'fc_hidden': 32,
    'n_classes': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 9)
    k = c['kernel_size']
    x = jax.random.uniform(ks[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(ks[1], (c['conv1_channels'], c['in_channels'], k, k), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['conv1_channels'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['conv2_channels'], c['conv1_channels'], k, k), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['conv2_channels'],), dtype) * 0.02
    wf1 = jax.random.normal(ks[5], (c['conv2_channels'], c['fc_hidden']), dtype) * 0.02
    bf1 = jax.random.normal(ks[6], (c['fc_hidden'],), dtype) * 0.02
    wf2 = jax.random.normal(ks[7], (c['fc_hidden'], c['n_classes']), dtype) * 0.02
    bf2 = jax.random.normal(ks[8], (c['n_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2, wf1, bf1, wf2, bf2


def _conv(x, w, b):
    # x NHWC, w in PyTorch OIHW layout, valid padding, stride 1.
    y = lax.conv_general_dilated(x, w, (1, 1), 'VALID',
                                 dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return y + b


def _maxpool(x):
    p = CONFIG['pool_size']
    # VALID padding == PyTorch floor mode (odd trailing row/col dropped).
    return lax.reduce_window(x, jnp.array(-jnp.inf, x.dtype), lax.max,
                             (1, p, p, 1), (1, p, p, 1), 'VALID')


def workload(x, w1, b1, w2, b2, wf1, bf1, wf2, bf2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_maxpool(_conv(h, w1, b1)))
    h = jax.nn.relu(_maxpool(_conv(h, w2, b2)))
    # PyTorch view(-1, 8) flattens the contiguous NCHW buffer, so go back to NCHW first.
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, wf1.shape[0])
    h = jax.nn.relu(h @ wf1 + bf1)
    return h @ wf2 + bf2
