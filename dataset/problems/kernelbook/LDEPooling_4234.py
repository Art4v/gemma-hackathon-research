"""LDE pooling: residual to dictionary, softmax over codewords of scaled squared distance, mean over frames."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LDEPooling_4234',
    'batch': 16,
    'input_dim': 128,
    'frames': 8192,
    'c_num': 64,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_mu = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['input_dim'], CONFIG['frames']), dtype)
    mu = jax.random.normal(k_mu, (CONFIG['input_dim'], CONFIG['c_num']), dtype) * 0.02
    s = jnp.ones((CONFIG['c_num'],), dtype)
    return x, mu, s


def workload(x, mu, s):
    b, d, _ = x.shape
    c = mu.shape[1]
    r = jnp.transpose(x, (0, 2, 1))[..., None] - mu  # (B, T, D, C)
    logits = -(s ** 2 + CONFIG['eps']) * jnp.sum(r ** 2, axis=2, keepdims=True)  # (B, T, 1, C)
    w = jax.nn.softmax(logits, axis=3)
    e = jnp.mean(w * r, axis=1)  # (B, D, C)
    return e.reshape(b, d * c, 1)
