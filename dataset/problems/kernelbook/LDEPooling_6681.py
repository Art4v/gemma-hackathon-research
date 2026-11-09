"""[bf16 variant] LDEPooling: residual to dictionary centers -> scaled squared distance -> softmax over centers -> weighted mean over frames."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LDEPooling_6681',
    'batch': 8,
    'input_dim': 128,
    'frames': 16384,
    'c_num': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_mu = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['input_dim'], CONFIG['frames']), dtype)
    mu = jax.random.normal(k_mu, (CONFIG['input_dim'], CONFIG['c_num']), dtype) * 0.02
    s = jnp.ones((CONFIG['c_num'],), dtype)
    return x, mu, s


def workload(x, mu, s):
    d, c = mu.shape
    r = jnp.transpose(x, (0, 2, 1))[..., None] - mu  # (B, T, D, C)
    w = jax.nn.softmax(s * jnp.sum(r ** 2, axis=2, keepdims=True), axis=3)  # (B, T, 1, C)
    e = jnp.mean(w * r, axis=1)  # (B, D, C)
    return e.reshape(-1, d * c, 1)
