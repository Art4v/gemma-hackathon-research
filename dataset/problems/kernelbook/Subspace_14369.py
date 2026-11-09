"""Subspace: (L * z) weighted sum of basis U over latent dim, plus mean mu."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Subspace_14369',
    'batch': 8,
    'latent_dim': 128,
    'channels': 16,
    'resolution': 128,
    'l_scale': 3.0,
}


def create_inputs(dtype=jnp.float32):
    b, d = CONFIG['batch'], CONFIG['latent_dim']
    c, r = CONFIG['channels'], CONFIG['resolution']
    k_z, k_u = jax.random.split(jax.random.key(0))
    z = jax.random.uniform(k_z, (b, d), dtype)
    # torch init: L = [[3*i for i in range(latent_dim, 0, -1)]]
    L = (CONFIG['l_scale'] * jnp.arange(d, 0, -1, dtype=jnp.float32)).astype(dtype)[None, :]
    U = jax.random.normal(k_u, (d, c, r, r), dtype) * 0.02
    mu = jnp.zeros((1, c, r, r), dtype)
    return z, L, U, mu


def workload(z, L, U, mu):
    # Uses the same broadcasting as the torch forward, so odd z ranks behave like torch.
    x = (L * z)[:, :, None, None, None]
    x = jnp.sum(U[None, ...] * x, axis=1)
    return x + mu
