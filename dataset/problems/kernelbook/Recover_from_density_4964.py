"""Nearest-neighbour upsample of lr_img (nn.Upsample, mode='nearest') followed by elementwise multiply with x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Recover_from_density_4964',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'upscale_factor': 1.0,
}


def create_inputs(dtype=jnp.float32):
    s = CONFIG['upscale_factor']
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    kx, kl = jax.random.split(jax.random.key(0))
    lr_img = jax.random.uniform(kl, (n, c, h, w), dtype)
    x = jax.random.uniform(kx, (n, c, int(h * s), int(w * s)), dtype)
    return x, lr_img


def _nearest_index(in_size, out_size, s):
    # PyTorch legacy 'nearest': src = floor(dst * (1/scale)), clamped to in_size-1.
    src = jnp.floor(jnp.arange(out_size, dtype=jnp.float32) * (1.0 / s)).astype(jnp.int32)
    return jnp.minimum(src, in_size - 1)


def workload(x, lr_img):
    s = CONFIG['upscale_factor']
    h, w = lr_img.shape[2], lr_img.shape[3]
    out_h, out_w = int(h * s), int(w * s)
    up = jnp.take(lr_img, _nearest_index(h, out_h, s), axis=2)
    up = jnp.take(up, _nearest_index(w, out_w, s), axis=3)
    return x * up
