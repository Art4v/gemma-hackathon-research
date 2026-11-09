"""[bf16 variant] ConvTranspose2d followed by channel min, height sum, GELU, and bias addition."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '36_ConvTranspose2d_Min_Sum_GELU_Add',
    'batch_size': 32,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'bias_shape': (1, 1, 1),
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    
    x = jax.random.uniform(k1, (B, C_in, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k4, CONFIG['bias_shape'], dtype=dtype) * 0.02
    return x, weight, conv_bias, bias

def workload(x, weight, conv_bias, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    K = weight.shape[2]
    
    pad_before = K - 1 - padding
    pad_after = K - 1 - padding + output_padding
    jax_padding = ((pad_before, pad_after), (pad_before, pad_after))
    
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    
    out_nhwc = lax.conv_transpose(
        x_nhwc,
        w_hwio,
        strides=(stride, stride),
        padding=jax_padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        transpose_kernel=True,
    )
    out_nhwc = out_nhwc + conv_bias
    x = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    
    x = jnp.min(x, axis=1, keepdims=True)
    x = jnp.sum(x, axis=2, keepdims=True)
    x = jax.nn.gelu(x, approximate=False)
    x = x + bias
    return x
