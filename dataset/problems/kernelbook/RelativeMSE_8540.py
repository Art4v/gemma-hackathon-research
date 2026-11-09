"""[bf16 variant] Relative MSE: 0.5 * mean((im - ref)^2 / (ref^2 + eps))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RelativeMSE_8540',
    'batch': 8,
    'channels': 16,
    'height': 512,
    'width': 512,
    'eps': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    k_im, k_ref = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    im = jax.random.uniform(k_im, shape, dtype)
    ref = jax.random.uniform(k_ref, shape, dtype)
    return im, ref


def workload(im, ref):
    loss = jnp.square(im - ref) / (jnp.square(ref) + CONFIG['eps'])
    return 0.5 * jnp.mean(loss)
