"""[m1 M2_second_output variant of kb/l2/58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp] ConvTranspose3d -> LogSumExp(channel) -> HardSwish-like -> subtract bias -> clamp, also returning the per-voxel channel max of the conv output (the logsumexp stabilizer)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp_m1_aux',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'bias_shape': (1, 1, 1, 1),
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        k0, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(k1, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(k2, (c['out_channels'],), dtype) * 0.02
    sub_bias = jax.random.normal(k3, c['bias_shape'], dtype) * 0.02
    return x, w, b, sub_bias


def workload(x, w, b, sub_bias):
    k = w.shape[2]
    s = CONFIG['stride']
    p = CONFIG['padding']
    pad = k - 1 - p
    x_nd = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC
    # PyTorch ConvTranspose weight (Cin, Cout, kd, kh, kw) -> flipped DHWIO
    w_k = jnp.transpose(w, (2, 3, 4, 0, 1))[::-1, ::-1, ::-1, :, :]
    y = lax.conv_general_dilated(
        x_nd, w_k,
        window_strides=(1, 1, 1),
        padding=[(pad, pad)] * 3,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    ch_max = jnp.transpose(jnp.max(y, axis=-1, keepdims=True), (0, 4, 1, 2, 3))  # N,1,D,H,W
    y = jax.nn.logsumexp(y, axis=-1, keepdims=True)
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # N,1,D,H,W
    y = y * jax.nn.sigmoid(y + 3.0) / 6.0
    y = y - sub_bias
    y = jnp.clip(y, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return y, ch_max
