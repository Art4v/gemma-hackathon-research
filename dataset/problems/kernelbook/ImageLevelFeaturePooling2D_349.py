"""[bf16 variant] ImageLevelFeaturePooling2D: view(2B, C, -1) -> mean(dim=2) -> view(-1, oc, 1, 1) -> nearest interpolate to (H, W)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ImageLevelFeaturePooling2D_349',
    'batch': 8,
    'channels': 512,
    'height': 128,
    'width': 128,
    'out_channels': 4,
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
    b, c, h, w = x.shape
    oc = CONFIG['out_channels']
    # Row-major reshape matches torch .view on a contiguous tensor.
    pooled = jnp.mean(x.reshape(b * 2, c, -1), axis=2)
    pooled = pooled.reshape(-1, oc, 1, 1)
    # Nearest-neighbour upsampling of a 1x1 map is a broadcast.
    return jnp.broadcast_to(pooled, (pooled.shape[0], oc, h, w))
