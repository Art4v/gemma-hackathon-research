"""Nearest-neighbour 1D resize (F.interpolate) followed by Conv1d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResizeConv1d_132',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'length': 32768,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'scale_factor': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['length']), dtype)
    # Weight layout (K, I, O) = torch (O, I, K) permuted to WIO.
    w = jax.random.normal(
        k_w, (CONFIG['kernel_size'], CONFIG['in_channels'], CONFIG['out_channels']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, w, b


def _nearest_resize(x, scale_factor):
    # Matches PyTorch 'nearest': src = min(floor(dst * (1/scale)), L_in - 1).
    l_in = x.shape[-1]
    l_out = int(l_in * scale_factor)
    src = jnp.floor(jnp.arange(l_out, dtype=jnp.float32) * (1.0 / scale_factor))
    src = jnp.minimum(src.astype(jnp.int32), l_in - 1)
    return jnp.take(x, src, axis=-1)


def workload(x, w, b):
    x = _nearest_resize(x, CONFIG['scale_factor'])
    x = jnp.transpose(x, (0, 2, 1))  # NCL -> NLC
    pad = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(CONFIG['stride'],),
        padding=((pad, pad),),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 2, 1))  # NLC -> NCL
