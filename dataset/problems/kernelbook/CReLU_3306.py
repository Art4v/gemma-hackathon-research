"""[bf16 variant] CReLU: cat(leaky_relu(x), leaky_relu(-leaky_relu(x))) along channels (PyTorch in-place aliasing)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CReLU_3306',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'negative_slope': 0.01,
    'concat_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    slope = CONFIG['negative_slope']
    # The PyTorch module's first in-place leaky_relu overwrites x, so the second
    # branch sees -leaky_relu(x), not -x.
    y = jax.nn.leaky_relu(x, negative_slope=slope)
    z = jax.nn.leaky_relu(-y, negative_slope=slope)
    return jnp.concatenate((y, z), axis=CONFIG['concat_axis'])
