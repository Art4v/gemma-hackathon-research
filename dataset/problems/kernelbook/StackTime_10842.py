"""[bf16 variant] StackTime: zero-padded time shifts -> concatenate on axis 2 -> subsample time by factor -> ceil-divide lengths."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'StackTime_10842',
    'time_steps': 256,
    'batch': 8,
    'channels': 128,
    'features': 128,
    'factor': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['time_steps'], CONFIG['batch'], CONFIG['channels'], CONFIG['features'])
    key = jax.random.key(0)
    k_x, k_lens = jax.random.split(key)
    x = jax.random.uniform(k_x, shape, dtype)
    x_lens = jax.random.randint(k_lens, shape, 1, CONFIG['time_steps'] + 1, dtype=jnp.int32)
    return x, x_lens


def workload(x, x_lens):
    factor = CONFIG['factor']
    seq = [x]
    for i in range(1, factor):
        seq.append(jnp.zeros_like(x).at[:-i].set(x[i:]))
    stacked = jnp.concatenate(seq, axis=2)[::factor]
    lens = jnp.ceil(x_lens.astype(jnp.float32) / factor).astype(jnp.int32)
    return stacked, lens
