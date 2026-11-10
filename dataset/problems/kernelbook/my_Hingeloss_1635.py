"""Hinge loss: relu(max_d2((1 - t) * o) - sum_d2(o * t) + 1), then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'my_Hingeloss_1635',
    'shape': (16, 64, 256, 512),
    'reduce_axis': 2,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.float32):
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    output = jax.random.uniform(k_out, CONFIG['shape'], dtype)
    target = jax.random.uniform(k_tgt, CONFIG['shape'], dtype)
    return output, target


def workload(output, target):
    axis = CONFIG['reduce_axis']
    pos = jnp.sum(output * target, axis=axis)
    neg = jnp.max((1 - target) * output, axis=axis)
    loss = jnp.maximum(neg - pos + CONFIG['margin'], 0)
    return jnp.mean(loss)
