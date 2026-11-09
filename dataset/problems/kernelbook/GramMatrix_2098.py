"""GramMatrix: reshape (B,C,H,W)->(B,C,HW), bmm(F, F^T), divide by H*W."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GramMatrix_2098',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w)
    # HIGHEST keeps the f32 contraction faithful to torch.bmm on TPU.
    g = jnp.einsum('bik,bjk->bij', f, f, precision=lax.Precision.HIGHEST)
    return g / (h * w)
