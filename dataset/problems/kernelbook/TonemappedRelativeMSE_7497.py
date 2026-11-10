"""[bf16 variant] Tonemapped relative MSE: clamp(min=0) -> Reinhard x/(1+x) on im and ref -> (im-ref)^2 / (ref^2+eps) -> 0.5*mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TonemappedRelativeMSE_7497',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 256,
    'eps': 0.01,
    'clamp_min': 0.0,
    'tonemap_offset': 1.0,
    'loss_scale': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_im, k_ref = jax.random.split(jax.random.key(0))
    im = jax.random.uniform(k_im, shape, dtype)
    ref = jax.random.uniform(k_ref, shape, dtype)
    return im, ref


def _tonemap(x):
    x = jnp.maximum(x, CONFIG['clamp_min'])
    return x / (CONFIG['tonemap_offset'] + x)


def workload(im, ref):
    im = _tonemap(im)
    ref = _tonemap(ref)
    mse = jnp.square(im - ref)
    loss = mse / (jnp.square(ref) + CONFIG['eps'])
    return CONFIG['loss_scale'] * jnp.mean(loss)
