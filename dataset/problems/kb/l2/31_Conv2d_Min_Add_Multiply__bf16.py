"""[bf16 variant] Conv2d -> min(x, constant) -> + bias -> * scaling_factor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '31_Conv2d_Min_Add_Multiply',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'constant_value': 0.5,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb, kbb = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    Cin = CONFIG['in_channels']
    Cout = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, Cin, H, W), dtype)
    conv_w = jax.random.normal(kw, (Cout, Cin, K, K), dtype) * 0.02
    conv_b = jax.random.normal(kb, (Cout,), dtype) * 0.02
    bias = jax.random.normal(kbb, (Cout, 1, 1), dtype) * 0.02
    return x, conv_w, conv_b, bias


def workload(x, conv_w, conv_b, bias):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + conv_b
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jnp.minimum(y, jnp.asarray(CONFIG['constant_value'], y.dtype))
    y = y + bias
    y = y * jnp.asarray(CONFIG['scaling_factor'], y.dtype)
    return y
