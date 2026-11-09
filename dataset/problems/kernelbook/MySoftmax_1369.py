"""Per-sample softmax over the channel axis (implicit dim=0 of each 3-D slice, i.e. axis 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MySoftmax_1369',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 256,
    'softmax_axis': 1,
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
    # F.softmax on 3-D input_[i] without dim defaults to dim 0 -> axis 1 of the batched tensor.
    return jax.nn.softmax(x, axis=CONFIG['softmax_axis'])
