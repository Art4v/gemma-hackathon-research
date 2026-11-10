"""GEGLU: Linear(dim_in -> 2*dim_out) -> chunk(2) -> x * gelu(gate) (exact erf GELU)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GEGLU_11635',
    'batch': 8,
    'seq_len': 2048,
    'dim_in': 1024,
    'dim_out': 4096,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim_in']), dtype)
    w = jax.random.normal(k_w, (CONFIG['dim_in'], 2 * CONFIG['dim_out']), dtype) * 0.02
    b = jax.random.normal(k_b, (2 * CONFIG['dim_out'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    h = x @ w + b
    val, gate = jnp.split(h, 2, axis=-1)
    return val * jax.nn.gelu(gate, approximate=False)
