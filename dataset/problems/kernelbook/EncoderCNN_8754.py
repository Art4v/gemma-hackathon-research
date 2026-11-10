"""[bf16 variant] EncoderCNN_8754: parallel dilated Conv2d + ELU branches -> concat -> 3x (Conv2d + ELU) -> 2x2 avg pool -> squeeze."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'EncoderCNN_8754',
    'batch': 8,
    'in_channels': 8,
    'image_size': 256,
    'branch_channels': 8,
    'dilations': (1, 2, 4, 8, 16),
    'paddings': (1, 3, 6, 12, 24),
    'conv2_channels': 256,
    'conv3_channels': 512,
    'latent_dim': 1024,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'base_dilation': 1,
    'pool_size': 2,
    'elu_alpha': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    num_branches = len(c['dilations'])
    channels = [(c['branch_channels'], c['in_channels'])] * num_branches + [
        (c['conv2_channels'], c['branch_channels'] * num_branches),
        (c['conv3_channels'], c['conv2_channels']),
        (c['latent_dim'], c['conv3_channels']),
    ]
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(channels))
    x = jax.random.uniform(
        keys[0],
        (c['batch'], c['in_channels'], c['image_size'], c['image_size']),
        dtype,
    )
    params = []
    for i, (out_ch, in_ch) in enumerate(channels):
        params.append(jax.random.normal(keys[1 + 2 * i], (out_ch, in_ch, k, k), dtype) * c['weight_scale'])
        params.append(jax.random.normal(keys[2 + 2 * i], (out_ch,), dtype) * c['weight_scale'])
    return (x, *params)


def _conv(x, w, b, stride, padding, dilation):
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(*inputs):
    c = CONFIG
    x, *p = inputs
    alpha = c['elu_alpha']
    num_branches = len(c['dilations'])
    x = jnp.transpose(x, (0, 2, 3, 1))
    branches = [
        jax.nn.elu(_conv(x, p[2 * i], p[2 * i + 1], c['stride'], pad, dil), alpha)
        for i, (dil, pad) in enumerate(zip(c['dilations'], c['paddings']))
    ]
    h = jnp.concatenate(branches, axis=-1)
    for i in range(num_branches, len(p) // 2):
        h = jax.nn.elu(_conv(h, p[2 * i], p[2 * i + 1], c['stride'], c['padding'], c['base_dilation']), alpha)
    w = c['pool_size']
    h = lax.reduce_window(h, 0.0, lax.add, (1, w, w, 1), (1, w, w, 1), 'VALID') / (w * w)
    h = jnp.transpose(h, (0, 3, 1, 2))
    return jnp.squeeze(h)
