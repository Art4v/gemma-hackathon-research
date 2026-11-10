"""[bf16 variant] Normalize3D: flatten per sample, per-sample min/max, min-max scale with eps, reshape to 3D."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Normalize3D_1787',
    'batch_size': 64,
    'num_features': 512,
    'num_steps': 1024,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['num_features'], CONFIG['num_steps']),
        dtype,
    )
    return (x,)


def workload(x):
    batch_size, num_features, num_steps = x.shape
    flat = x.reshape(batch_size, num_features * num_steps)
    max_value = jax.lax.stop_gradient(jnp.max(flat, axis=1, keepdims=True))
    min_value = jax.lax.stop_gradient(jnp.min(flat, axis=1, keepdims=True))
    scaled = (flat - min_value) / (max_value - min_value + CONFIG['eps'])
    return scaled.reshape(batch_size, num_features, num_steps)
