"""NormedLinear: L2-normalize weight rows and x along dim 1, scale by temperature, then linear + bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NormedLinear_5111',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'in_features': 512,
    'out_features': 1024,
    'temperature': 20.0,
    'power': 1.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    b = jnp.zeros((CONFIG['out_features'],), dtype)
    return x, w, b


def workload(x, w, b):
    power, eps = CONFIG['power'], CONFIG['eps']
    # w is (in, out): torch's per-row norm over in_features is axis 0 here.
    w_norm = jnp.sqrt(jnp.sum(w * w, axis=0, keepdims=True))
    w_ = w / (w_norm ** power + eps)
    # Matches the original: x is normalized over dim 1, not the feature axis.
    x_norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    x_ = x / (x_norm ** power + eps) * CONFIG['temperature']
    return x_ @ w_ + b
