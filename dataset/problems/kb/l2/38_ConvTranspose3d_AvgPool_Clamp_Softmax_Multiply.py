"""3D average pooling, transposed convolution, clamping, spatial softmax, and scaling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'pool_kernel_size': 2,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
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
    weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    scale = jnp.ones((1, C_out, 1, 1, 1), dtype=dtype)
    return (x, weight, bias, scale)


def workload(x, weight, bias, scale):
    pk = CONFIG['pool_kernel_size']
    K = CONFIG['kernel_size']
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    out_pad = CONFIG['output_padding']
    clamp_min = CONFIG['clamp_min']
    clamp_max = CONFIG['clamp_max']

    # Convert NCDHW -> NDHWC
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))

    # AvgPool3d
    window_shape = (1, pk, pk, pk, 1)
    x_pooled = lax.reduce_window(
        x_ndhwc,
        jnp.array(0.0, dtype=x.dtype),
        lax.add,
        window_dimensions=window_shape,
        window_strides=window_shape,
        padding='VALID',
    ) / (pk * pk * pk)

    # ConvTranspose3d via conv_general_dilated
    # weight is (C_in, C_out, KD, KH, KW) -> transpose to (KD, KH, KW, C_out, C_in) and flip spatial dims
    w_dhwio = jnp.transpose( jnp.flip(weight, axis=(2, 3, 4)), (2, 3, 4, 1, 0) )
    pad_before = K - 1 - pad
    pad_after = K - 1 - pad + out_pad
    conv_padding = ((pad_before, pad_after), (pad_before, pad_after), (pad_before, pad_after))

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_pooled,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=conv_padding,
        lhs_dilation=(stride, stride, stride),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    out_ndhwc = out_ndhwc + bias.reshape(1, 1, 1, 1, -1)

    # Convert NDHWC -> NCDHW
    out = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    # Clamp
    out = jnp.clip(out, clamp_min, clamp_max)

    # Spatial softmax
    b, c, d, h, w = out.shape
    out = out.reshape(b, c, -1)
    out = jax.nn.softmax(out, axis=2)
    out = out.reshape(b, c, d, h, w)

    # Multiply by learnable scale
    out = out * scale
    return out
