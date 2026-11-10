"""ActQuant_PACT: PACT clip to [0, |alpha|] then k-bit uniform quantization (round(x/alpha * (2^k-1)) / (2^k-1))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ActQuant_PACT_6801',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'act_bit': 4,
    'scale_coef': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    scale_coef = jnp.ones((1,), dtype=dtype) * CONFIG['scale_coef']
    return x, scale_coef


def workload(x, scale_coef):
    a_abs = jnp.abs(scale_coef)
    out = 0.5 * (jnp.abs(x) - jnp.abs(x - a_abs) + a_abs)
    # torch divides by the signed scale_coef here, not its abs.
    y = out / scale_coef
    n = float(2 ** CONFIG['act_bit'] - 1)
    return jnp.round(y * n) / n
