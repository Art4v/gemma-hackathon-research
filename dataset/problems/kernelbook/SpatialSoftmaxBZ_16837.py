"""[bf16 variant] Spatial softmax over flattened HxW followed by expected (x, y) keypoint coordinates."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SpatialSoftmaxBZ_16837',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    feature = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    # Same as np.meshgrid(linspace(-1,1,height), linspace(-1,1,width)) with 'xy' indexing.
    pos_x, pos_y = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, CONFIG['height']),
        jnp.linspace(-1.0, 1.0, CONFIG['width']))
    return feature, pos_x.reshape(-1).astype(dtype), pos_y.reshape(-1).astype(dtype)


def workload(feature, pos_x, pos_y):
    b, c = feature.shape[0], feature.shape[1]
    flattened = feature.reshape(b, c, -1)
    softmax = jax.nn.softmax(flattened, axis=-1)
    expected_x = jnp.sum(pos_y * softmax, axis=-1)
    expected_x = (-expected_x + 1.0) / 2.0
    expected_y = jnp.sum(pos_x * softmax, axis=-1)
    return jnp.stack([expected_x, expected_y], axis=2)
