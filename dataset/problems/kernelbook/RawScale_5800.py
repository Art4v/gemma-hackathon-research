"""RawScale: global quantile bounds -> clamp -> divide by max(|lower|, |upper|)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RawScale_5800',
    'shape': (8, 1536, 1024),
    'quantile': 1.0,
    'quantile_lower': 0.0,  # 0.5 - quantile / 2
    'quantile_upper': 1.0,  # 0.5 + quantile / 2
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    raw = jax.random.uniform(key, CONFIG['shape'], dtype)
    return (raw,)


def workload(raw):
    # ponytail: with quantile=1 these are min/max; jnp.quantile kept to match torch.quantile for any quantile.
    lower = jnp.quantile(raw, CONFIG['quantile_lower'], method='linear')
    upper = jnp.quantile(raw, CONFIG['quantile_upper'], method='linear')
    scale = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
    return jnp.clip(raw, lower, upper) / scale
