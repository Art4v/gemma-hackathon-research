"""[bf16 variant] GRU-style gate: sigmoid(Linear(cat[x, xh])) -> split r,u -> tanh(Linear(cat[x, r*xh])) -> u*z + (1-u)*xh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GRU_8485',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'outfea': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    b, d1, d2, f = CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['outfea']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(k[0], (b, d1, d2, f), dtype)
    xh = jax.random.uniform(k[1], (b, d1, d2, f), dtype)
    w_ff = jax.random.normal(k[2], (2 * f, 2 * f), dtype) * 0.02
    b_ff = jax.random.normal(k[3], (2 * f,), dtype) * 0.02
    w_zff = jax.random.normal(k[4], (2 * f, f), dtype) * 0.02
    b_zff = jax.random.normal(k[5], (f,), dtype) * 0.02
    return x, xh, w_ff, b_ff, w_zff, b_zff


def workload(x, xh, w_ff, b_ff, w_zff, b_zff):
    f = xh.shape[-1]
    gates = jax.nn.sigmoid(jnp.concatenate([x, xh], axis=-1) @ w_ff + b_ff)
    r, u = gates[..., :f], gates[..., f:]
    z = jnp.tanh(jnp.concatenate([x, r * xh], axis=-1) @ w_zff + b_zff)
    return u * z + (1 - u) * xh
