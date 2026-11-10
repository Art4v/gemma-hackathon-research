"""[bf16 variant] Sinkhorn_Net: view(-1,1) -> Linear(1,latent) -> ReLU -> Linear(latent,out) -> reshape(-1,out,out); dropout is identity."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Sinkhorn_Net_16387',
    'x_shape': (8, 128, 128, 128),
    'latent_dim': 32,
    'output_dim': 32,
    'dropout_prob': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    latent = CONFIG['latent_dim']
    out = CONFIG['output_dim']
    x = jax.random.uniform(kx, CONFIG['x_shape'], dtype)
    w1 = jax.random.normal(k1, (1, latent), dtype) * 0.02
    b1 = jax.random.normal(kb1, (latent,), dtype) * 0.02
    w2 = jax.random.normal(k2, (latent, out), dtype) * 0.02
    b2 = jax.random.normal(kb2, (out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    out = w2.shape[1]
    h = x.reshape(-1, 1)
    h = jax.nn.relu(h @ w1 + b1)
    h = h @ w2 + b2
    return h.reshape(-1, out, out)
