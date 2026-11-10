"""Gumbel-softmax hard sampler: double-log noise transform, tempered softmax, straight-through hard mask, argmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gumbel_sampler_16342',
    'batch': 8192,
    'num_classes': 2048,
    'eps': 1e-20,
    'temperature': 0.5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['num_classes'])
    key_input, key_noise = jax.random.split(jax.random.key(0))
    logits = jax.random.uniform(key_input, shape, dtype)
    noise = jax.random.uniform(key_noise, shape, dtype)
    return logits, noise


def workload(logits, noise):
    eps = CONFIG['eps']
    gumbel = -jnp.log(noise + eps)
    gumbel = -jnp.log(gumbel + eps)
    y = (logits + gumbel) / CONFIG['temperature']
    y = jax.nn.softmax(y, axis=-1)
    max_val = jnp.max(y, axis=-1, keepdims=True)
    max_idx = jnp.argmax(y, axis=-1)
    y_hard = (y == max_val).astype(y.dtype)
    y = jax.lax.stop_gradient(y_hard - y) + y
    return y, max_idx.reshape(1, -1)
