"""[bf16 variant] Decoder_2926: per-channel shared Linear (x[i] @ W + b for i < n_chan), stack on dim 0, transpose(0, 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Decoder_2926',
    'n_chan': 8,
    'batch': 16,
    'seq': 256,
    'h_dim': 1024,
    'out_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['n_chan'], c['batch'], c['seq'], c['h_dim']), dtype)
    w = jax.random.normal(kw, (c['h_dim'], c['out_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    y = jnp.einsum('...h,ho->...o', x[:CONFIG['n_chan']], w) + b
    return jnp.swapaxes(y, 0, 1)
