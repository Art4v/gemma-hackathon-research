"""Cat: split x along channels into size-2 chunks, take the first, concatenate it back onto x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Cat_4702',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'split_size': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    # torch.split(x, s, dim=1)[0] is x[:, :min(s, C)]
    size = min(CONFIG['split_size'], x.shape[1])
    addition = lax.slice_in_dim(x, 0, size, axis=1)
    return jnp.concatenate([x, addition], axis=1)
