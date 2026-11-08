"""3D transposed convolution with asymmetric input and kernel, strided, padded, and grouped."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'depth': 24,
    'height': 48,
    'width': 96,
    'kernel_size': (3, 5, 7),
    'stride': (2, 2, 2),
    'padding': (1, 2, 3),
    'output_padding': (1, 1, 1),
    'groups': 4,
}

def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    kD, kH, kW = CONFIG['kernel_size']
    G = CONFIG['groups']
    
    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out // G, kD, kH, kW), dtype=dtype) * 0.02
    return (x, weight)

def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    groups = CONFIG['groups']
    
    C_in = weight.shape[0]
    C_out_per_g = weight.shape[1]
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    C_in_per_g = C_in // groups
    C_out = C_out_per_g * groups
    
    # Convert NCDHW -> NDHWC
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    
    # PyTorch ConvTranspose3d weight has shape (in_channels, out_channels // groups, kD, kH, kW).
    # For lax.conv_general_dilated with feature_group_count=groups and rhs spec 'DHWIO',
    # rhs must have shape (kD, kH, kW, in_channels // groups, out_channels) and flipped spatial dims.
    w = jnp.reshape(weight, (groups, C_in_per_g, C_out_per_g, kD, kH, kW))
    w = jnp.flip(w, axis=(3, 4, 5))
    w = jnp.transpose(w, (3, 4, 5, 1, 0, 2))
    w_dhwio = jnp.reshape(w, (kD, kH, kW, C_in_per_g, C_out))
    
    pad_d = (kD - 1 - padding[0], kD - 1 - padding[0] + output_padding[0])
    pad_h = (kH - 1 - padding[1], kH - 1 - padding[1] + output_padding[1])
    pad_w = (kW - 1 - padding[2], kW - 1 - padding[2] + output_padding[2])
    jax_padding = [pad_d, pad_h, pad_w]
    
    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=jax_padding,
        lhs_dilation=stride,
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=groups
    )
    
    # Convert NDHWC -> NCDHW
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
