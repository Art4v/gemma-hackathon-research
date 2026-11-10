"""[bf16 variant] SiamFC fast cross-correlation: grouped conv2d of search x with templates z, times out_scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SiamFC_5438',
    'nz': 16,
    'nx': 16,
    'channels': 256,
    'z_size': 16,
    'x_size': 128,
    'out_scale': 0.001,
}


def create_inputs(dtype=jnp.bfloat16):
    kz, kx = jax.random.split(jax.random.key(0))
    c = CONFIG['channels']
    z = jax.random.uniform(kz, (CONFIG['nz'], c, CONFIG['z_size'], CONFIG['z_size']), dtype)
    x = jax.random.uniform(kx, (CONFIG['nx'], c, CONFIG['x_size'], CONFIG['x_size']), dtype)
    return z, x


def workload(z, x):
    nz = z.shape[0]
    nx, c, h, w = x.shape
    xg = x.reshape(-1, nz * c, h, w)
    xg = jnp.transpose(xg, (0, 2, 3, 1))
    kernel = jnp.transpose(z, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        xg, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=nz,
        precision=lax.Precision.HIGHEST,
    )
    out = jnp.transpose(out, (0, 3, 1, 2))
    out = out.reshape(nx, -1, out.shape[-2], out.shape[-1])
    return out * CONFIG['out_scale']
