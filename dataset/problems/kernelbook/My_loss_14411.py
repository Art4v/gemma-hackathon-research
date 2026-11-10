"""Per-slice concordance correlation coefficient (mean, Pearson rho, ddof=1 std) over axis 1, summed and negated."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'My_loss_14411',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'ddof': 1,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_x, shape, dtype)
    y = jax.random.uniform(key_y, shape, dtype)
    return x, y


def workload(x, y):
    ddof = CONFIG['ddof']

    def ccc(x_i, y_i):
        x_m = jnp.mean(x_i)
        y_m = jnp.mean(y_i)
        vx = x_i - x_m
        vy = y_i - y_m
        rho = jnp.sum(vx * vy) / (jnp.sqrt(jnp.sum(vx ** 2)) * jnp.sqrt(jnp.sum(vy ** 2)))
        x_s = jnp.std(x_i, ddof=ddof)
        y_s = jnp.std(y_i, ddof=ddof)
        return 2 * rho * x_s * y_s / (x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2)

    n = x.shape[-1]
    cccs = jax.vmap(ccc, in_axes=(1, 1))(x[:, :n], y[:, :n])
    return -jnp.sum(cccs)
