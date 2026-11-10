"""Per-row min-max audio normalization: max/min over axis 1, (x - min) / (max - min + eps) * 2 - 1."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Normalize_1783',
    'batch_size': 128,
    'audio_length': 262144,
    'eps': 1e-10,
    'scale': 2.0,
    'shift': 1.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    audio = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['audio_length']), dtype
    )
    return (audio,)


def workload(audio):
    max_value = lax.stop_gradient(jnp.max(audio, axis=1, keepdims=True))
    min_value = lax.stop_gradient(jnp.min(audio, axis=1, keepdims=True))
    normalized = (audio - min_value) / (max_value - min_value + CONFIG['eps'])
    return normalized * CONFIG['scale'] - CONFIG['shift']
