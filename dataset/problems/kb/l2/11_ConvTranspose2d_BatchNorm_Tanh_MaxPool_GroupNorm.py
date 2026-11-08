"""ConvTranspose2d -> BatchNorm2d (batch stats) -> Tanh -> MaxPool2d(2,2) -> GroupNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm',
    'batch_size': 512,
    'in_channels': 64,
    'out_channels': 128,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'stride': 1,
    'padding': 1,
    'groups': 8,
    'num_groups': 8,
    'pool_kernel': 2,
    'pool_stride': 2,
    'bn_eps': 1e-5,
    'gn_eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    bn_w = jnp.ones((c['out_channels'],), dtype)
    bn_b = jnp.zeros((c['out_channels'],), dtype)
    gn_w = jnp.ones((c['out_channels'],), dtype)
    gn_b = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, bn_w, bn_b, gn_w, gn_b


def workload(x, w, b, bn_w, bn_b, gn_w, gn_b):
    c = CONFIG
    K = w.shape[2]
    s = c['stride']
    p = c['padding']
    # NCHW -> NHWC
    h = jnp.transpose(x, (0, 2, 3, 1))
    # PyTorch ConvTranspose weight (Cin, Cout, K, K) -> flipped HWIO (K, K, Cin, Cout)
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    pad = K - 1 - p
    h = lax.conv_general_dilated(
        h, w_hwio,
        window_strides=(1, 1),
        padding=[(pad, pad), (pad, pad)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    h = h + b
    # BatchNorm with batch statistics
    mean = jnp.mean(h, axis=(0, 1, 2), keepdims=True)
    var = jnp.var(h, axis=(0, 1, 2), keepdims=True)
    h = (h - mean) * lax.rsqrt(var + c['bn_eps']) * bn_w + bn_b
    h = jnp.tanh(h)
    # MaxPool
    pk, ps = c['pool_kernel'], c['pool_stride']
    h = lax.reduce_window(h, -jnp.inf, lax.max, (1, pk, pk, 1), (1, ps, ps, 1), 'VALID')
    # GroupNorm
    N, H, W, C = h.shape
    G = c['num_groups']
    hg = h.reshape(N, H, W, G, C // G)
    gmean = jnp.mean(hg, axis=(1, 2, 4), keepdims=True)
    gvar = jnp.var(hg, axis=(1, 2, 4), keepdims=True)
    hg = (hg - gmean) * lax.rsqrt(gvar + c['gn_eps'])
    h = hg.reshape(N, H, W, C) * gn_w + gn_b
    # NHWC -> NCHW
    return jnp.transpose(h, (0, 3, 1, 2))
