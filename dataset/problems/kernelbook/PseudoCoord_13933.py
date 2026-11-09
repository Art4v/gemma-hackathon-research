"""[bf16 variant] PseudoCoord: box centers -> pairwise relative coords -> (rho, theta) polar pseudo-coordinates."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PseudoCoord_13933',
    'batch': 24576,
    'num_obj': 128,
    'box_dim': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    b = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['num_obj'], CONFIG['box_dim']), dtype=dtype
    )
    return (b,)


def workload(b):
    centers = (b[:, :, 2:] + b[:, :, :2]) * 0.5
    rel = centers[:, :, None, :] - centers[:, None, :, :]
    dx = rel[..., 0]
    dy = rel[..., 1]
    rho = jnp.sqrt(dx ** 2 + dy ** 2)
    theta = jnp.arctan2(dx, dy)
    return jnp.stack((rho, theta), axis=-1)
