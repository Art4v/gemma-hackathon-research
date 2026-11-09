"""[bf16 variant] Conv2d(k=10, s=2) + bias -> ReLU -> NCHW flatten -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SimpleCNN32Filter_12057',
    'batch': 16384,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'out_channels': 32,
    'kernel_size': 10,
    'stride': 2,
    'num_classes': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    oh = (c['height'] - c['kernel_size']) // c['stride'] + 1
    ow = (c['width'] - c['kernel_size']) // c['stride'] + 1
    flat = c['out_channels'] * oh * ow
    x = jax.random.uniform(k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    conv_w = jax.random.normal(
        k[1], (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(k[2], (c['out_channels'],), dtype) * 0.02
    fc_w = jax.random.normal(k[3], (flat, c['num_classes']), dtype) * 0.02
    fc_b = jax.random.normal(k[4], (c['num_classes'],), dtype) * 0.02
    return x, conv_w, conv_b, fc_w, fc_b


def workload(x, conv_w, conv_b, fc_w, fc_b):
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, conv_w, window_strides=(s, s), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    y = jax.nn.relu(y + conv_b)
    # Flatten in PyTorch's (C, H, W) order to match x.view(-1, 144 * 32).
    y = jnp.transpose(y, (0, 3, 1, 2)).reshape(-1, fc_w.shape[0])
    return y @ fc_w + fc_b
