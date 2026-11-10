"""Log STFT magnitude loss: log(x_mag), log(y_mag), mean absolute difference."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LogSTFTMagnitudeLoss_13455',
    'batch': 8,
    'channels': 128,
    'frames': 256,
    'freq_bins': 128,
    'min_mag': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['frames'], CONFIG['freq_bins'])
    key = jax.random.key(0)
    k_x, k_y = jax.random.split(key)
    x_mag = jax.random.uniform(k_x, shape, dtype, minval=CONFIG['min_mag'], maxval=1.0)
    y_mag = jax.random.uniform(k_y, shape, dtype, minval=CONFIG['min_mag'], maxval=1.0)
    return x_mag, y_mag


def workload(x_mag, y_mag):
    return jnp.mean(jnp.abs(jnp.log(y_mag) - jnp.log(x_mag)))
