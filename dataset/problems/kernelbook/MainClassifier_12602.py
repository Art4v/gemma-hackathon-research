"""[bf16 variant] AdaptiveAvgPool2d(1) -> flatten -> Linear (MainClassifier)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MainClassifier_12602',
    'batch': 64,
    'channel': 512,
    'height': 32,
    'width': 32,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channel'], CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (CONFIG['channel'], CONFIG['num_classes']), dtype) * 0.02
    b = jax.random.normal(kb, (CONFIG['num_classes'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3))  # (B, C), same as AdaptiveAvgPool2d(1) + view
    return pooled @ w + b
