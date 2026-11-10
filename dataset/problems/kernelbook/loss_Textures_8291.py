"""[bf16 variant] loss_Textures: reshape to (B, -1, nc, H, W), sum of squares over nc, relu(alpha * yi2 - xi2 + margin), mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'loss_Textures_8291',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'nc': 1,
    'alpha': 1.2,
    'margin': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_x, k_y = jax.random.split(key)
    x = jax.random.uniform(k_x, shape, dtype)
    y = jax.random.uniform(k_y, shape, dtype)
    return x, y


def workload(x, y):
    nc = CONFIG['nc']
    xi = jnp.reshape(x, (x.shape[0], -1, nc, x.shape[2], x.shape[3]))
    yi = jnp.reshape(y, (y.shape[0], -1, nc, y.shape[2], y.shape[3]))
    xi2 = jnp.sum(xi * xi, axis=2)
    yi2 = jnp.sum(yi * yi, axis=2)
    out = jnp.maximum(yi2 * CONFIG['alpha'] - xi2 + CONFIG['margin'], 0.0)
    return jnp.mean(out)
