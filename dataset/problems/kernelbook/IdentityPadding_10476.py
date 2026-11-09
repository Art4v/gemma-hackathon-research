"""[bf16 variant] IdentityPadding: channel zero-pad (F.pad) followed by MaxPool2d(kernel=1, stride)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'IdentityPadding_10476',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'stride': 1,
    'add_channels': 0,  # out_channels - in_channels, same as the torch module
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    add = CONFIG['add_channels']
    s = CONFIG['stride']
    out = jnp.pad(x, ((0, 0), (0, add), (0, 0), (0, 0)))
    # ponytail: MaxPool2d with kernel 1 and no padding is a strided slice
    return out[:, :, ::s, ::s]
