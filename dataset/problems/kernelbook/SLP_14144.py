"""SLP: view(-1, input_size) -> Linear -> log_softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SLP_14144',
    'x_shape': (16, 128, 128, 128),
    'input_size': 4096,
    'logits': 1024,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, CONFIG['x_shape'], dtype)
    w = jax.random.normal(kw, (CONFIG['input_size'], CONFIG['logits']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['logits'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    h = x.reshape(-1, w.shape[0]) @ w + b
    return jax.nn.log_softmax(h, axis=1)
