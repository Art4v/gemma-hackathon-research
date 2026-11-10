"""LayerNorm with unbiased std: mean, std(ddof=1).clamp(eps), normalize, affine."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_5644',
    'batch': 16,
    'seq_len': 2048,
    'hidden_size': 1024,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size']), dtype)
    weight = jnp.ones((CONFIG['hidden_size'],), dtype)
    bias = jnp.zeros((CONFIG['hidden_size'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    # torch.std defaults to the unbiased estimator (ddof=1).
    sigma = jnp.maximum(jnp.std(x, axis=-1, keepdims=True, ddof=1), CONFIG['eps'])
    return (x - mu) / sigma * weight + bias
