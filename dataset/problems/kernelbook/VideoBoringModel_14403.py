"""AdaptiveAvgPool3d(1) over (D, H, W), squeeze, Linear to out_features."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VideoBoringModel_14403',
    'batch': 8,
    'in_channel': 128,
    'depth': 8,
    'height': 128,
    'width': 128,
    'out_features': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channel'], CONFIG['depth'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['in_channel'], CONFIG['out_features']), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(-3, -2, -1), keepdims=True)
    features = jnp.squeeze(pooled)
    return features @ w + b
