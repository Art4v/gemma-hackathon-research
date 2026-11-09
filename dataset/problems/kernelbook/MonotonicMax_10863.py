"""MonotonicMax: per-sample max over H, stacked so the output is (C, B, W)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MonotonicMax_10863',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
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
    # torch: cat over the batch of max(x[b], dim=1).unsqueeze(1) -> (C, B, W)
    return jnp.transpose(jnp.max(x, axis=2), (1, 0, 2))
