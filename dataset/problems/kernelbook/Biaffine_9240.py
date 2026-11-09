"""[bf16 variant] Biaffine: concat ones bias to x and y, then einsum bxi,oij,byj->boxy, squeeze n_out."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Biaffine_9240',
    'batch': 16,
    'len_x': 1024,
    'len_y': 1024,
    'n_in': 1024,
    'n_out': 1,
    'bias_x': True,
    'bias_y': True,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky, kw = jax.random.split(jax.random.key(0), 3)
    b, n_in = CONFIG['batch'], CONFIG['n_in']
    x = jax.random.uniform(kx, (b, CONFIG['len_x'], n_in), dtype)
    y = jax.random.uniform(ky, (b, CONFIG['len_y'], n_in), dtype)
    w = jax.random.normal(
        kw,
        (CONFIG['n_out'], n_in + int(CONFIG['bias_x']), n_in + int(CONFIG['bias_y'])),
        dtype,
    ) * 0.02
    return x, y, w


def _append_ones(a):
    return jnp.concatenate([a, jnp.ones_like(a[..., :1])], axis=-1)


def workload(x, y, weight):
    if CONFIG['bias_x']:
        x = _append_ones(x)
    if CONFIG['bias_y']:
        y = _append_ones(y)
    s = jnp.einsum('bxi,oij,byj->boxy', x, weight, y,
                   precision=lax.Precision.HIGHEST)
    if s.shape[1] == 1:
        s = jnp.squeeze(s, axis=1)
    return s
