"""3D convolution, division, max pooling, global average pooling, bias addition, and sum reduction."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum',
    'batch_size': 64,
    'in_channels': 16,
    'out_channels': 32,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_d': 3,
    'kernel_h': 3,
    'kernel_w': 3,
    'divisor': 2.0,
    'pool_d': 2,
    'pool_h': 2,
    'pool_w': 2,
    'sum_dim': 1,
}

def create_inputs(dtype=jnp.float32):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(
        k0,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['depth'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    conv_weight = jax.random.normal(
        k1,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_d'], CONFIG['kernel_h'], CONFIG['kernel_w']),
        dtype=dtype,
    ) * 0.02
    conv_bias = jax.random.normal(k2, (CONFIG['out_channels'],), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (CONFIG['out_channels'], 1, 1, 1), dtype=dtype) * 0.02
    return x, conv_weight, conv_bias, bias

def workload(x, conv_weight, conv_bias, bias):
    # Convert NCDHW -> NDHWC
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # Convert OIdhw -> dhwIO
    w_dhwio = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
    
    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape, w_dhwio.shape, ('NDHWC', 'DHWIO', 'NDHWC')
    )
    out = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )
    out = out + conv_bias.reshape(1, 1, 1, 1, -1)
    out = out / CONFIG['divisor']
    
    pool_window = (1, CONFIG['pool_d'], CONFIG['pool_h'], CONFIG['pool_w'], 1)
    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=pool_window,
        window_strides=pool_window,
        padding='VALID',
    )
    
    # Global average pooling over spatial dimensions (D, H, W)
    out = jnp.mean(out, axis=(1, 2, 3), keepdims=True)
    
    # Convert NDHWC -> NCDHW
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    
    out = out + bias
    out = jnp.sum(out, axis=CONFIG['sum_dim'])
    return out
