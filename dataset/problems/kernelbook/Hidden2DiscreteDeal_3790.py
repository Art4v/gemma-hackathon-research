"""[bf16 variant] Hidden2DiscreteDeal: linear projection on the last axis (x @ w + b) followed by log_softmax, returning (logits, log_pz)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hidden2DiscreteDeal_3790',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_size': 512,
    'z_size': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['input_size']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['input_size'], CONFIG['z_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['z_size'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    logits = x @ w + b
    log_pz = jax.nn.log_softmax(logits, axis=-1)
    return logits, log_pz
