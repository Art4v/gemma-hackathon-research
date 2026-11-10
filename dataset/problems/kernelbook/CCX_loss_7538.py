"""[bf16 variant] CCX contextual loss: center by y mean, L2-normalize channels, batched cosine matmul, relative distance, exp softmax-style affinity, max/mean/log reduction."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CCX_loss_7538',
    'batch': 8,
    'channels': 1024,
    'height': 32,
    'width': 32,
    'eps': 1e-06,
    'h': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, ky = jax.random.split(key)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def workload(x, y):
    eps = CONFIG['eps']
    h = CONFIG['h']
    N, C = x.shape[0], x.shape[1]
    y_mu = y.mean(axis=3).mean(axis=2).mean(axis=0).reshape(1, -1, 1, 1)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / jnp.linalg.norm(x_centered, ord=2, axis=1, keepdims=True)
    y_normalized = y_centered / jnp.linalg.norm(y_centered, ord=2, axis=1, keepdims=True)
    x_normalized = x_normalized.reshape(N, C, -1)
    y_normalized = y_normalized.reshape(N, C, -1)
    cosine_sim = jnp.matmul(jnp.transpose(x_normalized, (0, 2, 1)), y_normalized)
    d = 1 - cosine_sim
    d_min = jnp.min(d, axis=2, keepdims=True)
    d_tilde = d / (d_min + eps)
    w = jnp.exp((1 - d_tilde) / h)
    ccx_ij = w / jnp.sum(w, axis=2, keepdims=True)
    ccx = jnp.mean(jnp.max(ccx_ij, axis=1), axis=1)
    ccx_loss = jnp.mean(-jnp.log(ccx + eps))
    return ccx_loss
