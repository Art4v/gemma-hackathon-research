"""[bf16 variant] 3D transposed convolution followed by two 3D max pooling layers and channel-wise sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '78_ConvTranspose3d_Max_Max_Sum',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'stride': 2,
    'padding': 2,
    'pool1_kernel': 2,
    'pool2_kernel': 3,
}

def create_inputs(dtype=jnp.bfloat16):
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
    weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)

def _max_pool3d(x_ndhwc, k):
    B, D, H, W, C = x_ndhwc.shape
    d_out, h_out, w_out = D // k, H // k, W // k
    x_cropped = x_ndhwc[:, :d_out * k, :h_out * k, :w_out * k, :]
    x_reshaped = x_cropped.reshape(B, d_out, k, h_out, k, w_out, k, C)
    return jnp.max(x_reshaped, axis=(2, 4, 6))

def workload(x, weight, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    K = weight.shape[2]
    p1 = CONFIG['pool1_kernel']
    p2 = CONFIG['pool2_kernel']

    # NCDHW -> NDHWC
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # PyTorch weight: (C_in, C_out, kD, kH, kW) -> DHWIO: (kD, kH, kW, C_out, C_in)
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))

    pad = K - 1 - padding
    jax_padding = ((pad, pad), (pad, pad), (pad, pad))

    x_ndhwc = lax.conv_transpose(
        x_ndhwc,
        w_dhwio,
        strides=(stride, stride, stride),
        padding=jax_padding,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        transpose_kernel=True,
    )
    x_ndhwc = x_ndhwc + bias.reshape(1, 1, 1, 1, -1)

    x_ndhwc = _max_pool3d(x_ndhwc, p1)
    x_ndhwc = _max_pool3d(x_ndhwc, p2)

    # Sum over channels (axis -1 in NDHWC) and convert to NCDHW with channel dim = 1
    out = jnp.sum(x_ndhwc, axis=-1, keepdims=True)
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    return out
