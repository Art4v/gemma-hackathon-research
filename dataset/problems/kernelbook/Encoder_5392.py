"""[bf16 variant] Encoder_5392: Conv1d (kernel L, stride L//2, no bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Encoder_5392',
    'batch': 8,
    'audio_channels': 4,
    'T': 1048576,
    'L': 16,
    'N': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['audio_channels'], CONFIG['T']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['N'], CONFIG['audio_channels'], CONFIG['L']), dtype) * 0.02
    return x, w


def workload(x, w):
    # PyTorch Conv1d accepts unbatched (C, T) input; mirror that.
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    L = w.shape[2]
    x_nwc = jnp.transpose(x, (0, 2, 1))
    w_wio = jnp.transpose(w, (2, 1, 0))
    y = lax.conv_general_dilated(
        x_nwc, w_wio,
        window_strides=(L // 2,),
        padding='VALID',
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = jnp.transpose(jax.nn.relu(y), (0, 2, 1))
    return y[0] if unbatched else y
