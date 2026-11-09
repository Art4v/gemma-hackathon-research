"""MaxPPVPool1d: concat(max over last axis, fraction of positives over last axis), then unsqueeze(2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaxPPVPool1d_4472',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'width': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    _max = jnp.max(x, axis=-1)
    _ppv = jnp.sum(x > 0, axis=-1).astype(jnp.float32) / x.shape[-1]
    out = jnp.concatenate((_max.astype(jnp.promote_types(x.dtype, jnp.float32)), _ppv), axis=-1)
    return jnp.expand_dims(out, 2)
