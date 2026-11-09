"""[bf16 variant] ConvTranspose2d -> min over channels -> sum over height -> GELU -> add bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '36_ConvTranspose2d_Min_Sum_GELU_Add',
    'batch_size': 16,
    'in_channels': 64,
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
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(key, 4)
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kbias, c['bias_shape'], dtype) * 0.02
    return x, w, b, bias


def workload(x, w, b, bias):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # NCHW -> NHWC
    xh = jnp.transpose(x, (0, 2, 3, 1))
    # PyTorch transposed-conv weight (Cin, Cout, kH, kW) -> flipped HWIO
    wk = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = jnp.transpose(y, (0, 3, 1, 2))  # back to NCHW
    y = jnp.min(y, axis=1, keepdims=True)
    y = jnp.sum(y, axis=2, keepdims=True)
    y = jax.nn.gelu(y, approximate=False)
    return y + bias
