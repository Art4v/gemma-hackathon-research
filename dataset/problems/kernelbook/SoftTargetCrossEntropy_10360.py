"""[bf16 variant] Soft-target cross entropy: log_softmax over last axis, weighted negative sum, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftTargetCrossEntropy_10360',
    'batch': 8,
    'height': 128,
    'width': 128,
    'classes': 256,
    'weights': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['classes'])
    key_x, key_target = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_x, shape, dtype)
    target = jax.random.uniform(key_target, shape, dtype)
    return x, target


def workload(x, target):
    loss = jnp.sum(-target * jax.nn.log_softmax(x, axis=-1) * CONFIG['weights'], axis=-1)
    return jnp.mean(loss)
