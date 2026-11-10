"""[bf16 variant] cSE: global average pool -> linear -> relu -> linear -> sigmoid -> channel-wise rescale of x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'cSE_2843',
    'batch': 8,
    'channels': 512,
    'reduced_channels': 256,
    'height': 128,
    'width': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w1, k_w2 = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(
        k_w1, (CONFIG['channels'], CONFIG['reduced_channels']), dtype
    ) * CONFIG['weight_scale']
    w2 = jax.random.normal(
        k_w2, (CONFIG['reduced_channels'], CONFIG['channels']), dtype
    ) * CONFIG['weight_scale']
    return x, w1, w2


def workload(x, w1, w2):
    y = jnp.mean(x, axis=(2, 3))
    y = y @ w1
    y = jax.nn.relu(y)
    y = y @ w2
    y = jax.nn.sigmoid(y)
    return x * y[:, :, None, None]
