"""VectorQuantizer: NCHW->NHWC, L2 distance to codebook, argmin, embedding gather, commitment loss, straight-through output."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'VectorQuantizer_17814',
    'batch': 16,
    'e_dim': 256,
    'height': 64,
    'width': 64,
    'n_e': 1024,
    'beta': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k_z, k_e = jax.random.split(jax.random.key(0))
    z = jax.random.uniform(
        k_z, (CONFIG['batch'], CONFIG['e_dim'], CONFIG['height'], CONFIG['width']), dtype)
    codebook = jax.random.normal(k_e, (CONFIG['n_e'], CONFIG['e_dim']), dtype) * 0.02
    return z, codebook


def workload(z, codebook):
    e_dim = codebook.shape[1]
    z = jnp.transpose(z, (0, 2, 3, 1))
    z_flat = z.reshape(-1, e_dim)
    d = (jnp.sum(z_flat ** 2, axis=1, keepdims=True)
         + jnp.sum(codebook ** 2, axis=1)[None, :]
         - 2 * jnp.matmul(z_flat, codebook.T, precision=lax.Precision.HIGHEST))
    idx = jnp.argmin(d, axis=1)
    z_q = jnp.take(codebook, idx, axis=0).reshape(z.shape)
    loss = (CONFIG['beta'] * jnp.mean((lax.stop_gradient(z_q) - z) ** 2)
            + jnp.mean((z_q - lax.stop_gradient(z)) ** 2))
    z_q = z + lax.stop_gradient(z_q - z)
    z_q = jnp.transpose(z_q, (0, 3, 1, 2))
    return z_q, loss, idx.reshape(z.shape[:-1])
