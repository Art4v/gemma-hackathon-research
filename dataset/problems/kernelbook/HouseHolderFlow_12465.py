"""HouseHolderFlow: z - 2 * v (v^T z) / ||v||^2 per row."""
import jax
import jax.numpy as jnp

CONFIG = {'name': 'HouseHolderFlow_12465', 'batch': 16384, 'latent': 2048}


def create_inputs(dtype=jnp.float32):
    kv, kz = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['latent'])
    v = jax.random.uniform(kv, shape, dtype)
    z = jax.random.uniform(kz, shape, dtype)
    return v, z


def workload(v, z):
    # ponytail: (v v^T) z == v (v^T z); avoids materializing the B x L x L outer product.
    vz = jnp.sum(v * z, axis=1, keepdims=True)
    norm_sq = jnp.sum(v * v, axis=1, keepdims=True)
    return z - 2 * (v * vz) / norm_sq
