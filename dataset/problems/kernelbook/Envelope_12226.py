"""[bf16 variant] Envelope: where(x < 1, 1/x + a*x^p + b*x^(p+1) + c*x^(p+2), 0)."""
import jax
import jax.numpy as jnp

_P = 4

CONFIG = {
    'name': 'Envelope_12226',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'exponent': _P,
    'a': -(_P + 1) * (_P + 2) / 2,
    'b': float(_P * (_P + 2)),
    'c': -_P * (_P + 1) / 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    p, a, b, c = CONFIG['exponent'], CONFIG['a'], CONFIG['b'], CONFIG['c']
    x_pow_p0 = x ** p
    x_pow_p1 = x_pow_p0 * x
    env_val = 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x
    return jnp.where(x < 1, env_val, jnp.zeros_like(x))
