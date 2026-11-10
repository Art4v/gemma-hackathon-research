"""merge_layer: mean over last dim -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'merge_layer_17261',
    'batch': 8,
    'seq': 128,
    'hidden_size': 1024,
    'reduce_dim': 64,
    'output_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx,
        (CONFIG['batch'], CONFIG['seq'], CONFIG['hidden_size'], CONFIG['reduce_dim']),
        dtype=dtype,
    )
    w = jax.random.normal(kw, (CONFIG['hidden_size'], CONFIG['output_size']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['output_size'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    h = jnp.mean(x, axis=-1)
    return jnp.tanh(h @ w + b)
