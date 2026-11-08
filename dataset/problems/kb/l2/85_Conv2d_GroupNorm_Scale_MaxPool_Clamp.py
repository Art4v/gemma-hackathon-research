"""Conv2d followed by GroupNorm, channel-wise scaling, MaxPool2d, and clamping."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '85_Conv2d_GroupNorm_Scale_MaxPool_Clamp',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'num_groups': 16,
    'maxpool_kernel_size': 4,
    'eps': 1e-5,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    N = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (N, C_in, H, W), dtype=dtype)
    conv_w = jax.random.normal(k2, (C_out, C_in, K, K), dtype=dtype) * 0.02
    conv_b = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    gn_w = jnp.ones((C_out,), dtype=dtype)
    gn_b = jnp.zeros((C_out,), dtype=dtype)
    scale = jnp.ones((C_out, 1, 1), dtype=dtype)
    return x, conv_w, conv_b, gn_w, gn_b, scale


def workload(x, conv_w, conv_b, gn_w, gn_b, scale):
    N, _, _, _ = x.shape
    C_out = conv_w.shape[0]
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    mp_k = CONFIG['maxpool_kernel_size']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out_nhwc = out_nhwc + jnp.reshape(conv_b, (1, 1, 1, C_out))
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))

    _, _, H_out, W_out = out.shape
    out_grouped = jnp.reshape(out, (N, G, C_out // G, H_out, W_out))
    mean = jnp.mean(out_grouped, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(out_grouped, axis=(2, 3, 4), keepdims=True)
    out_norm = (out_grouped - mean) * lax.rsqrt(var + eps)
    out = jnp.reshape(out_norm, (N, C_out, H_out, W_out))
    out = out * jnp.reshape(gn_w, (1, C_out, 1, 1)) + jnp.reshape(gn_b, (1, C_out, 1, 1))

    out = out * scale

    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, mp_k, mp_k),
        window_strides=(1, 1, mp_k, mp_k),
        padding='VALID',
    )

    out = jnp.clip(out, CONFIG['clamp_min'], CONFIG['clamp_max'])
    return out
