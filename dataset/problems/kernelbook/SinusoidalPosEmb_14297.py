"""SinusoidalPosEmb: exp frequency ramp -> x * freqs broadcast -> concat(sin, cos)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SinusoidalPosEmb_14297',
    'batch': 256,
    'seq_len': 65536,
    'dim': 32,
    'log_max_period': 9.210340371976184,  # math.log(10000) as a double
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['seq_len']), dtype)
    steps = jnp.arange(CONFIG['dim'] // 2, dtype=dtype)
    return x, steps


def workload(x, steps):
    scale = CONFIG['log_max_period'] / (steps.shape[0] - 1)
    freqs = jnp.exp(steps * -scale)
    emb = x[:, :, None] * freqs[None, :]
    return jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
