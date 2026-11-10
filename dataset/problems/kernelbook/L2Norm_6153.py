"""L2Norm_6153: sqrt(sum(x*x, axis=1) + eps) -> unsqueeze(-1) -> broadcast to x -> x / norm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2Norm_6153',
    'batch': 16,
    'size': 128,  # channels == height == width; expand_as requires the last three dims to match
    'sum_axis': 1,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['size'], CONFIG['size'], CONFIG['size'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    norm = jnp.sqrt(jnp.sum(x * x, axis=CONFIG['sum_axis']) + CONFIG['eps'])
    # Matches norm.unsqueeze(-1).expand_as(x): the new axis is appended last, not at sum_axis.
    expanded = jnp.broadcast_to(norm[..., None], x.shape)
    return x / expanded
