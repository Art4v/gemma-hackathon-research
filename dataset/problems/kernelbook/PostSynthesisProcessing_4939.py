"""PostSynthesisProcessing: shift, scale to [0, 255], add rounding offset, clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PostSynthesisProcessing_4939',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'min_value': -1,
    'max_value': 1,
    'scale': 255.0,
    'offset': 0.5,
    'clamp_min': 0,
    'clamp_max': 255,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    synthesized_image = jax.random.uniform(key, shape, dtype)
    return (synthesized_image,)


def workload(synthesized_image):
    min_value = CONFIG['min_value']
    max_value = CONFIG['max_value']
    scaled = (synthesized_image - min_value) * CONFIG['scale'] / (max_value - min_value)
    return jnp.clip(scaled + CONFIG['offset'], CONFIG['clamp_min'], CONFIG['clamp_max'])
