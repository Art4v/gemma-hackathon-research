"""Rodrigues: channel L2 norm with eps, normalize, cos/sin, stack 9 rotation terms, reshape to (-1, 3, 3)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Rodrigues_16825',
    'batch': 32,
    'channels': 4,
    'height': 512,
    'width': 512,
    'eps': 1e-05,
    'matrix_side': 3,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    rvec = jax.random.uniform(key, shape, dtype)
    return (rvec,)


def workload(rvec):
    theta = jnp.sqrt(CONFIG['eps'] + jnp.sum(rvec ** 2, axis=1))
    r = rvec / theta[:, None]
    costh = jnp.cos(theta)
    sinth = jnp.sin(theta)
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    v = 1.0 - costh
    out = jnp.stack(
        (
            x ** 2 + (1.0 - x ** 2) * costh,
            x * y * v - z * sinth,
            x * z * v + y * sinth,
            x * y * v + z * sinth,
            y ** 2 + (1.0 - y ** 2) * costh,
            y * z * v - x * sinth,
            x * z * v - y * sinth,
            y * z * v + x * sinth,
            z ** 2 + (1.0 - z ** 2) * costh,
        ),
        axis=1,
    )
    side = CONFIG['matrix_side']
    return out.reshape(-1, side, side)
