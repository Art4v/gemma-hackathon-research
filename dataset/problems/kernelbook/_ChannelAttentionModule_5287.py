"""[bf16 variant] Channel attention: bmm(f, f^T) -> (rowmax - att) -> softmax -> bmm(att, f) -> beta * e + x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '_ChannelAttentionModule_5287',
    'batch': 8,
    'channels': 512,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    # ponytail: zeros matches the torch init of beta; output == x for these inputs.
    beta = jnp.zeros((1,), dtype)
    return x, beta


def workload(x, beta):
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w)
    att = jnp.einsum('bcn,bdn->bcd', f, f, precision=lax.Precision.HIGHEST)
    att = jnp.max(att, axis=-1, keepdims=True) - att
    att = jax.nn.softmax(att, axis=-1)
    e = jnp.einsum('bcd,bdn->bcn', att, f, precision=lax.Precision.HIGHEST)
    return beta * e.reshape(b, c, h, w) + x
