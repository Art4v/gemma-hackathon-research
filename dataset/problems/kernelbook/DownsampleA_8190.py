"""DownsampleA: AvgPool2d(kernel=1, stride=2) then channel concat with zeros (x.mul(0))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DownsampleA_8190',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'stride': 2,
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
    s = CONFIG['stride']
    # AvgPool2d with kernel_size=1 averages a single element, so it is a strided slice.
    y = x[:, :, ::s, ::s]
    # y * 0 (not zeros) keeps PyTorch's NaN/inf propagation from x.mul(0).
    return jnp.concatenate([y, y * 0], axis=1)
