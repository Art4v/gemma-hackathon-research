"""L1 loss: optional bilinear resize of fake to real's size, then mean(abs(scale*real - scale*fake))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L1_12240',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'scale': 10.0,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_fake, k_real = jax.random.split(jax.random.key(0))
    fake = jax.random.uniform(k_fake, shape, dtype)
    real = jax.random.uniform(k_real, shape, dtype)
    return fake, real


def workload(fake, real):
    if fake.shape != real.shape:
        H, W = real.shape[2], real.shape[3]
        fake = jax.image.resize(fake, fake.shape[:2] + (H, W), method='bilinear', antialias=False)
    scale = CONFIG['scale']
    return jnp.mean(jnp.abs(scale * real - scale * fake))
