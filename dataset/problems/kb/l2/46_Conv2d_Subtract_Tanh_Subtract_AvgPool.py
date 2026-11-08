"""Conv2d followed by scalar subtraction, tanh, scalar subtraction, and AvgPool2d."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '46_Conv2d_Subtract_Tanh_Subtract_AvgPool',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'subtract1_value': 0.5,
    'subtract2_value': 0.2,
    'kernel_size_pool': 2,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    
    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)

def workload(x, weight, bias):
    k_pool = CONFIG['kernel_size_pool']
    sub1 = CONFIG['subtract1_value']
    sub2 = CONFIG['subtract2_value']
    
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    
    dn = lax.conv_dimension_numbers(x_nhwc.shape, w_hwio.shape, ('NHWC', 'HWIO', 'NHWC'))
    out = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )
    out = out + bias.reshape(1, 1, 1, -1)
    out = out - sub1
    out = jnp.tanh(out)
    out = out - sub2
    
    window_shape = (1, k_pool, k_pool, 1)
    out = lax.reduce_window(
        out,
        init_value=jnp.array(0, dtype=out.dtype),
        computation=lax.add,
        window_dimensions=window_shape,
        window_strides=window_shape,
        padding='VALID',
    )
    out = out / (k_pool * k_pool)
    
    return jnp.transpose(out, (0, 3, 1, 2))
