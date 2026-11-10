"""Hidden2Discrete: Linear -> view(-1, k_size) -> log_softmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Hidden2Discrete_684',
    'batch': 16384,
    'input_size': 1024,
    'y_size': 32,
    'k_size': 128,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['input_size']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['input_size'], CONFIG['y_size'], CONFIG['k_size']), dtype) * 0.02
    b = jnp.zeros((CONFIG['y_size'], CONFIG['k_size']), dtype)
    return x, w, b


def workload(x, w, b):
    # w is (in, y, k): k_size comes from the shape so torch-sized checks also match.
    in_size, y_size, k_size = w.shape
    logits = x @ w.reshape(in_size, y_size * k_size) + b.reshape(-1)
    logits = logits.reshape(-1, k_size)
    log_qy = jax.nn.log_softmax(logits, axis=1)
    return logits, log_qy
