"""[bf16 variant] PositionalEncoder_8622: x/100 -> unsqueeze -> scale by logspace bands * pi -> concat sin/cos -> flatten(1) -> left zero-pad."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PositionalEncoder_8622',
    'batch': 8,
    'height': 128,
    'width': 128,
    'depth': 128,
    'max_freq': 4.0,
    'feat_size': 68,
    'dimensionality': 4,
    'base': 2.0,
    'num_bands': 8,
    'pad': 4,
    'divisor': 100.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['depth'])
    x = jax.random.uniform(key, shape, dtype)
    end = jnp.log(CONFIG['max_freq'] / 2) / jnp.log(CONFIG['base'])
    scales = jnp.logspace(0.0, end, CONFIG['num_bands'], base=CONFIG['base'], dtype=dtype)
    return x, scales


def workload(x, scales):
    x = x / CONFIG['divisor']
    x = x[..., None]
    scales = scales[(None,) * (x.ndim - 1) + (Ellipsis,)]
    x = x * scales * jnp.pi
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    enc = x.reshape(x.shape[0], x.size // x.shape[0])
    return jnp.pad(enc, ((0, 0), (CONFIG['pad'], 0)))
