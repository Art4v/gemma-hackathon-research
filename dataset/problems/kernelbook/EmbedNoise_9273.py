"""EmbedNoise: Linear -> LeakyReLU -> view to (N, C, D, D, D)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EmbedNoise_9273',
    'z_shape': (8, 32, 32, 1024),
    'z_dim': 1024,
    'channels': 128,
    'dim': 4,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    kz, kw, kb = jax.random.split(jax.random.key(0), 3)
    out_features = CONFIG['channels'] * CONFIG['dim'] ** 3
    z = jax.random.uniform(kz, CONFIG['z_shape'], dtype)
    w = jax.random.normal(kw, (CONFIG['z_dim'], out_features), dtype) * 0.02
    b = jax.random.normal(kb, (out_features,), dtype) * 0.02
    return z, w, b


def workload(z, w, b):
    d = CONFIG['dim']
    channels = w.shape[1] // (d * d * d)
    out = jnp.matmul(z, w) + b
    out = jnp.where(out >= 0, out, CONFIG['negative_slope'] * out)
    return out.reshape((-1, channels, d, d, d))
