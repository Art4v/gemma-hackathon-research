"""[bf16 variant] AE_6834: view(-1,dim) -> Linear+ReLU -> Linear+ReLU (z) -> Linear+ReLU -> Linear+ReLU (recon); returns (recon, z)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AE_6834',
    'x_shape': (8, 16, 128, 1024),
    'dim': 1024,
    'hidden': 512,
    'latent': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    dim, hid, lat = CONFIG['dim'], CONFIG['hidden'], CONFIG['latent']
    kx, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, CONFIG['x_shape'], dtype)
    w1 = jax.random.normal(k1, (dim, hid), dtype) * 0.02
    b1 = jnp.zeros((hid,), dtype)
    w2 = jax.random.normal(k2, (hid, lat), dtype) * 0.02
    b2 = jnp.zeros((lat,), dtype)
    w3 = jax.random.normal(k3, (lat, hid), dtype) * 0.02
    b3 = jnp.zeros((hid,), dtype)
    w4 = jax.random.normal(k4, (hid, dim), dtype) * 0.02
    b4 = jnp.zeros((dim,), dtype)
    return x, w1, b1, w2, b2, w3, b3, w4, b4


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    dim = w1.shape[0]
    h = x.reshape(-1, dim)
    h1 = jax.nn.relu(h @ w1 + b1)
    z = jax.nn.relu(h1 @ w2 + b2)
    h3 = jax.nn.relu(z @ w3 + b3)
    recon = jax.nn.relu(h3 @ w4 + b4)
    return recon, z
