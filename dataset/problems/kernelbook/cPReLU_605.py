"""Complex PReLU: chunk along complex axis, PReLU on real and imag halves, concatenate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'cPReLU_605',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'complex_axis': 1,
    'num_parameters': 1,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.normal(jax.random.key(0), shape, dtype)
    r_a = jnp.full((CONFIG['num_parameters'],), CONFIG['prelu_init'], dtype)
    i_a = jnp.full((CONFIG['num_parameters'],), CONFIG['prelu_init'], dtype)
    return x, r_a, i_a


def workload(x, r_a, i_a):
    axis = CONFIG['complex_axis']
    split_at = (x.shape[axis] + 1) // 2
    real, imag = jnp.split(x, [split_at], axis=axis)
    real = jnp.where(real >= 0, real, r_a * real)
    imag = jnp.where(imag >= 0, imag, i_a * imag)
    return jnp.concatenate([real, imag], axis=axis)
