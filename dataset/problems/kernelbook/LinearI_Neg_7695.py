"""LinearI_Neg: y = x @ (-exp(W)) + b (linear layer with negated exponentiated weight)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearI_Neg_7695',
    'batch': 16,
    'dim1': 32,
    'dim2': 128,
    'in_features': 1024,
    'out_features': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # w is stored (in, out); exp is elementwise so transposing before or after is equivalent.
    return x @ (-jnp.exp(w)) + b
