"""ConvTranspose2d followed by GELU and GroupNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '19_ConvTranspose2d_GELU_GroupNorm',
    'batch_size': 32,
    'in_channels': 64,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'groups': 8,
    'num_groups': 8,
    'eps': 1e-5,
}

def create_inputs(dtype=jnp.float32):
    k0, k1 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k1, (C_in, C_out, K, K), dtype=dtype) * 0.02
    bias = jnp.zeros((C_out,), dtype=dtype)
    gn_weight = jnp.ones((C_out,), dtype=dtype)
    gn_bias = jnp.zeros((C_out,), dtype=dtype)
    return (x, weight, bias, gn_weight, gn_bias)

def workload(x, weight, bias, gn_weight, gn_bias):
    S = CONFIG['stride']
    K = CONFIG['kernel_size']
    G = CONFIG['num_groups']
    eps = CONFIG['eps']

    # NCHW -> NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # IOHW -> HWIO
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    pad = K - 1
    out_nhwc = lax.conv_transpose(
        x_nhwc,
        w_hwio,
        strides=(S, S),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        transpose_kernel=True,
    )
    out_nhwc = out_nhwc + bias.reshape(1, 1, 1, -1)
    # NHWC -> NCHW
    x = jnp.transpose(out_nhwc, (0, 3, 1, 2))

    # Exact GELU matching torch.nn.functional.gelu default
    x = jax.nn.gelu(x, approximate=False)

    # GroupNorm
    N, C, H_out, W_out = x.shape
    x = x.reshape(N, G, C // G, H_out, W_out)
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = x.reshape(N, C, H_out, W_out)
    x = x * gn_weight.reshape(1, C, 1, 1) + gn_bias.reshape(1, C, 1, 1)
    return x
