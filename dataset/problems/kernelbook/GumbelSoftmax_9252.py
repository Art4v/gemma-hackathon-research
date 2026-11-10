"""[bf16 variant] Linear projection, flatten to (-1, c_dim), then two softmaxes returning (logits, prob, y)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GumbelSoftmax_9252',
    'batch': 8,
    'height': 128,
    'width': 128,
    'f_dim': 512,
    'c_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['f_dim']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['f_dim'], CONFIG['c_dim']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['c_dim'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    c_dim = w.shape[-1]
    logits = jnp.reshape(x @ w + b, (-1, c_dim))
    prob = jax.nn.softmax(logits, axis=-1)
    y = jax.nn.softmax(logits, axis=-1)
    return logits, prob, y
