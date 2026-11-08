"""3D convolution followed by LeakyReLU, broadcast addition, clamp, and GELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '90_Conv3d_LeakyReLU_Sum_Clamp_GELU',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'sum_tensor_shape': (64, 1, 1, 1),
    'negative_slope': 0.2,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
}

def create_inputs(dtype=jnp.float32):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k1, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k2, (C_out,), dtype=dtype) * 0.02
    sum_tensor = jax.random.normal(k3, CONFIG['sum_tensor_shape'], dtype=dtype) * 0.02
    return (x, weight, bias, sum_tensor)

def workload(x, weight, bias, sum_tensor):
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape,
        w_dhwio.shape,
        ('NDHWC', 'DHWIO', 'NDHWC')
    )
    out = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=dn
    )
    out = out + bias.reshape(1, 1, 1, 1, -1)
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    out = jax.nn.leaky_relu(out, negative_slope=CONFIG['negative_slope'])
    out = out + sum_tensor
    out = jnp.clip(out, CONFIG['clamp_min'], CONFIG['clamp_max'])
    out = jax.nn.gelu(out, approximate=False)
    return out
