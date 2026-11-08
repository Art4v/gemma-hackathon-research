"""Conv3d followed by HardSwish, GroupNorm, and spatial Mean pooling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '27_Conv3d_HardSwish_GroupNorm_Mean',
    'batch_size': 1024,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 4,
    'num_groups': 4,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    gn_weight = jnp.ones((C_out,), dtype=dtype)
    gn_bias = jnp.zeros((C_out,), dtype=dtype)
    return x, conv_weight, conv_bias, gn_weight, gn_bias

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    B, C_in, D, H, W = x.shape
    C_out = conv_weight.shape[0]
    G = CONFIG['num_groups']
    eps = CONFIG['eps']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))

    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    out_ndhwc = out_ndhwc + conv_bias
    out = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    out = jax.nn.hard_swish(out)

    _, _, D_out, H_out, W_out = out.shape
    out_grouped = jnp.reshape(out, (B, G, C_out // G, D_out, H_out, W_out))
    mean = jnp.mean(out_grouped, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(out_grouped, axis=(2, 3, 4, 5), keepdims=True)
    out_norm = (out_grouped - mean) / jnp.sqrt(var + eps)
    out = jnp.reshape(out_norm, (B, C_out, D_out, H_out, W_out))
    out = out * jnp.reshape(gn_weight, (1, C_out, 1, 1, 1)) + jnp.reshape(gn_bias, (1, C_out, 1, 1, 1))

    out = jnp.mean(out, axis=(2, 3, 4))
    return out
