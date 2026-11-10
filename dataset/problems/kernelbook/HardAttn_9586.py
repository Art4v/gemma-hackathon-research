"""[bf16 variant] Global average pool over NCHW spatial axes -> linear -> tanh -> reshape to (N, num_points, coord_dim)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HardAttn_9586',
    'batch': 32,
    'in_channels': 128,
    'height': 128,
    'width': 128,
    'num_points': 4,
    'coord_dim': 2,
    'weight_scale': 0.02,
    'bias_init': (0.0, -0.75, 0.0, -0.25, 0.0, 0.25, 0.0, 0.75),
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['in_channels'], CONFIG['num_points'] * CONFIG['coord_dim']),
        dtype,
    ) * CONFIG['weight_scale']
    b = jnp.array(CONFIG['bias_init'], dtype=dtype)
    return x, w, b


def workload(x, w, b):
    pooled = jnp.mean(x, axis=(2, 3))
    theta = jnp.tanh(pooled @ w + b)
    return theta.reshape(-1, CONFIG['num_points'], CONFIG['coord_dim'])
