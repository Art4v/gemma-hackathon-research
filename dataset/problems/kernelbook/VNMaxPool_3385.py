"""[bf16 variant] VNMaxPool: channel linear map to direction, per-point dot product, argmax over points, gather x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'VNMaxPool_3385',
    'batch': 16,
    'channels': 128,
    'vec_dim': 3,
    'num_points': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['vec_dim'], CONFIG['num_points']),
        dtype,
    )
    w = jax.random.normal(kw, (CONFIG['channels'], CONFIG['channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    # w is (in, out): d[b,o,k,n] = sum_i x[b,i,k,n] * w[i,o]
    d = jnp.einsum('bikn,io->bokn', x, w, precision=lax.Precision.HIGHEST)
    dotprod = jnp.sum(x * d, axis=2)               # [B, C, N]
    idx = jnp.argmax(dotprod, axis=-1)             # [B, C]
    b, c, k, _ = x.shape
    gather_idx = jnp.broadcast_to(idx[:, :, None, None], (b, c, k, 1))
    return jnp.take_along_axis(x, gather_idx, axis=-1)[..., 0]  # [B, C, 3]
