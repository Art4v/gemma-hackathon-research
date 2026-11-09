"""[bf16 variant] ClusterLayer: mu = 1 / sum(|x[:, None] - center| ** (2 / (m - 1)), axis=2), normalized over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ClusterLayer_758',
    'batch': 32,
    'reduce_dim': 128,
    'n_cluster': 128,
    'expansion': 128,
    'cluster_m': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_c = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['reduce_dim'], CONFIG['n_cluster'], CONFIG['expansion']),
        dtype=dtype,
    )
    center = jax.random.normal(k_c, (CONFIG['n_cluster'], CONFIG['expansion']), dtype) * 0.02
    return x, center


def workload(x, center):
    p = 2.0 / (CONFIG['cluster_m'] - 1.0)
    # Same broadcasting as torch: center aligns with the trailing dims of x.unsqueeze(1).
    mu = 1.0 / jnp.sum(jnp.abs(x[:, None] - center) ** p, axis=2)
    return mu / jnp.sum(mu, axis=1, keepdims=True)
