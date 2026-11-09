"""[bf16 variant] Conv3d -> ReLU -> LeakyReLU -> GELU -> Sigmoid -> BiasAdd."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd',
    'batch_size': 64,
    'in_channels': 8,
    'out_channels': 32,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(k, 4)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    ks = c['kernel_size']
    w = jax.random.normal(kw, (ks, ks, ks, c['in_channels'], c['out_channels']), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kbias, (c['out_channels'], 1, 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1, 1), padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + conv_b
    y = jax.nn.relu(y)
    y = jnp.where(y >= 0, y, y * CONFIG['negative_slope'])
    y = jax.nn.gelu(y, approximate=False)
    y = jax.nn.sigmoid(y)
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
    return y + bias
