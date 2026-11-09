"""Conv1d over (N, T, C) with 'same' padding plus bias, last step dropped for an even kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv1d_8318',
    'batch': 16,
    'seq_len': 4096,
    'in_channels': 512,
    'out_channels': 512,
    'kernel_size': 4,
    'stride': 1,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['in_channels']), dtype)
    w = jax.random.normal(kw, (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    k = CONFIG['kernel_size']
    dil = CONFIG['dilation']
    pad = k // 2 * dil
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(CONFIG['stride'],),
        padding=((pad, pad),),
        rhs_dilation=(dil,),
        dimension_numbers=('NWC', 'OIW', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b[None, None, :]
    if k % 2 == 0:
        y = y[:, :-1, :]
    return y
