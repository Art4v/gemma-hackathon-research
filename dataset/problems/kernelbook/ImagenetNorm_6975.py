"""[bf16 variant] ImagenetNorm: x / 255 then per-channel (x - mean) / std on NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ImagenetNorm_6975',
    'batch': 16,
    'channels': 3,
    'height': 1024,
    'width': 1024,
    'from_raw': True,
    'raw_scale': 255.0,
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    mean = jnp.array(CONFIG['mean'], dtype=jnp.float32)
    std = jnp.array(CONFIG['std'], dtype=jnp.float32)
    return x, mean, std


def workload(x, mean, std):
    x = x.astype(jnp.float32)
    if CONFIG['from_raw']:
        x = x / CONFIG['raw_scale']
    c = mean.shape[0]
    return (x - mean.reshape(1, c, 1, 1)) / std.reshape(1, c, 1, 1)
