"""Conv2d followed by two Mish activations."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '4_Conv2d_Mish_Mish',
    'batch_size': 32,
    'in_channels': 64,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    k = CONFIG['kernel_size']
    
    x = jax.random.uniform(k1, (b, c_in, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_out, c_in, k, k), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    return (x, weight, bias)

def _mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

def workload(x, weight, bias):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=weight,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    out_nhwc = out_nhwc + bias
    out = jnp.transpose(out_nhwc, (0, 3, 1, 2))
    out = _mish(out)
    out = _mish(out)
    return out
