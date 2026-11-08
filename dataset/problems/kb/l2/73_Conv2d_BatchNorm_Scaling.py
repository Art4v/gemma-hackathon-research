"""Conv2d followed by BatchNorm2d and scalar multiplication."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '73_Conv2d_BatchNorm_Scaling',
    'batch_size': 64,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'scaling_factor': 2.0,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, H, W), dtype=dtype)
    conv_w = jax.random.normal(k1, (C_out, C_in, K, K), dtype=dtype) * 0.02
    conv_b = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    bn_gamma = jnp.ones((C_out,), dtype=dtype)
    bn_beta = jnp.zeros((C_out,), dtype=dtype)
    return (x, conv_w, conv_b, bn_gamma, bn_beta)

def workload(x, conv_w, conv_b, bn_gamma, bn_beta):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    dn = lax.conv_dimension_numbers(x_nhwc.shape, w_hwio.shape, ('NHWC', 'HWIO', 'NHWC'))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )
    out_nhwc = out_nhwc + conv_b.reshape(1, 1, 1, -1)
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))

    mean = jnp.mean(out, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(out - mean), axis=(0, 2, 3), keepdims=True)
    out = (out - mean) / jnp.sqrt(var + CONFIG['eps'])
    out = out * bn_gamma.reshape(1, -1, 1, 1) + bn_beta.reshape(1, -1, 1, 1)

    out = out * CONFIG['scaling_factor']
    return out
