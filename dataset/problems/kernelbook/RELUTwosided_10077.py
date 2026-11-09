"""[bf16 variant] RELUTwosided: relu(|x| - lam*sigma^2/L) * sign(x) with per-channel lam."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RELUTwosided_10077',
    'batch': 8,
    'num_conv': 128,
    'height': 256,
    'width': 256,
    'lam': 0.001,
    'L': 100,
    'sigma': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['num_conv'], CONFIG['height'], CONFIG['width']),
        dtype,
        minval=-1.0,
        maxval=1.0,
    )
    lam = jnp.full((1, CONFIG['num_conv'], 1, 1), CONFIG['lam'], dtype)
    return x, lam


def workload(x, lam):
    la = lam * CONFIG['sigma'] ** 2
    return jax.nn.relu(jnp.abs(x) - la / CONFIG['L']) * jnp.sign(x)
