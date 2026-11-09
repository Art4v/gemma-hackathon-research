"""[bf16 variant] ComplexConv1d: (conv_r(x_r) - conv_i(x_i), conv_r(x_i) + conv_i(x_r)) with two biased Conv1d layers."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexConv1d_16119',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'length': 16384,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 6)
    n, cin, cout = CONFIG['batch'], CONFIG['in_channels'], CONFIG['out_channels']
    length, k = CONFIG['length'], CONFIG['kernel_size']
    x_r = jax.random.uniform(keys[0], (n, cin, length), dtype)
    x_i = jax.random.uniform(keys[1], (n, cin, length), dtype)
    w_r = jax.random.normal(keys[2], (cout, cin, k), dtype) * 0.02
    b_r = jax.random.normal(keys[3], (cout,), dtype) * 0.02
    w_i = jax.random.normal(keys[4], (cout, cin, k), dtype) * 0.02
    b_i = jax.random.normal(keys[5], (cout,), dtype) * 0.02
    return x_r, x_i, w_r, b_r, w_i, b_i


def _conv1d(x_nwc, w_oik, b):
    w_wio = jnp.transpose(w_oik, (2, 1, 0))
    pad = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nwc, w_wio,
        window_strides=(CONFIG['stride'],),
        padding=[(pad, pad)],
        rhs_dilation=(CONFIG['dilation'],),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x_r, x_i, w_r, b_r, w_i, b_i):
    # torch Conv1d accepts unbatched (C, L) input; the KernelBench get_inputs uses that form.
    unbatched = x_r.ndim == 2
    if unbatched:
        x_r, x_i = x_r[None], x_i[None]
    xr = jnp.transpose(x_r, (0, 2, 1))
    xi = jnp.transpose(x_i, (0, 2, 1))
    out_r = _conv1d(xr, w_r, b_r) - _conv1d(xi, w_i, b_i)
    out_i = _conv1d(xi, w_r, b_r) + _conv1d(xr, w_i, b_i)
    out_r = jnp.transpose(out_r, (0, 2, 1))
    out_i = jnp.transpose(out_i, (0, 2, 1))
    if unbatched:
        out_r, out_i = out_r[0], out_i[0]
    return out_r, out_i
