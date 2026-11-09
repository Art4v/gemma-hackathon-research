"""[bf16 variant] Linear(latent_dim -> 1) followed by ReLU (NumPredictor)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NumPredictor_18261',
    'd0': 16,
    'd1': 64,
    'd2': 128,
    'latent_dim': 512,
    'out_features': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], CONFIG['latent_dim']), dtype)
    w = jax.random.normal(kw, (CONFIG['latent_dim'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    return jax.nn.relu(x @ w + b)
