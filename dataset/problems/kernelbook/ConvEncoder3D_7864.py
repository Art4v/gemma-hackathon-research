"""[bf16 variant] ConvEncoder3D: transpose -> 6x (Conv3d + ReLU) -> spatial mean -> ReLU -> Linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvEncoder3D_7864',
    'batch': 8,
    'seq': 16,
    'in_channels': 3,
    'height': 256,
    'width': 256,
    'hidden_dim': 32,
    'c_dim': 128,
    'kernel_size': 3,
    'padding': 1,
    'strides': ((1, 2, 2), (2, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    'channel_mults': (1, 2, 4, 8, 16, 16),
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    chans = [c['in_channels']] + [c['hidden_dim'] * m for m in c['channel_mults']]
    n_conv = len(c['channel_mults'])
    keys = jax.random.split(jax.random.key(0), 2 * n_conv + 3)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['seq'], c['in_channels'], c['height'], c['width']), dtype)
    params = []
    for i in range(n_conv):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, k, chans[i], chans[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    fc_w = jax.random.normal(keys[-2], (chans[-1], c['c_dim']), dtype) * 0.02
    fc_b = jax.random.normal(keys[-1], (c['c_dim'],), dtype) * 0.02
    return (x, *params, fc_w, fc_b)


def workload(x, *params):
    conv_params, fc_w, fc_b = params[:-2], params[-2], params[-1]
    p = CONFIG['padding']
    # (B, T, C, H, W) -> (B, T, H, W, C): equals torch transpose(1,2) then NCDHW->NDHWC.
    net = jnp.transpose(x, (0, 1, 3, 4, 2))
    for i, stride in enumerate(CONFIG['strides']):
        if i > 0:
            net = jax.nn.relu(net)
        w, b = conv_params[2 * i], conv_params[2 * i + 1]
        net = lax.conv_general_dilated(
            net, w, window_strides=stride, padding=[(p, p)] * 3,
            dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')) + b
    net = jnp.mean(net, axis=(1, 2, 3))
    return jax.nn.relu(net) @ fc_w + fc_b
