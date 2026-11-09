"""[bf16 variant] QuantConv2d: activation fake-quant (clamp(0.1x) to 8-bit) + tanh weight fake-quant (8-bit) -> conv2d + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'QuantConv2d_1261',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'a_bits': 8,
    'w_bits': 8,
    'act_scale': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _round(v):
    # Round half away from zero, matching the PyTorch Round Function (not banker's rounding).
    return jnp.sign(v) * jnp.floor(jnp.abs(v) + 0.5)


def _quant_act(x):
    scale = 1.0 / float(2 ** CONFIG['a_bits'] - 1)
    q = jnp.clip(x * CONFIG['act_scale'], 0.0, 1.0)
    return _round(q / scale) * scale


def _quant_weight(w):
    scale = 1.0 / float(2 ** CONFIG['w_bits'] - 1)
    t = jnp.tanh(w)
    t = t / 2 / jnp.max(jnp.abs(t)) + 0.5
    q = _round(t / scale) * scale
    return 2 * q - 1


def workload(x, w, b):
    c = CONFIG
    qx = jnp.transpose(_quant_act(x), (0, 2, 3, 1))           # NCHW -> NHWC
    qw = jnp.transpose(_quant_weight(w), (2, 3, 1, 0))        # OIHW -> HWIO
    p = c['padding']
    y = lax.conv_general_dilated(
        qx, qw,
        window_strides=(c['stride'], c['stride']),
        padding=((p, p), (p, p)),
        rhs_dilation=(c['dilation'], c['dilation']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))                     # NHWC -> NCHW
