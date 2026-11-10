"""[bf16 variant] AdjMSELoss: abs(outputs - labels) * exp(-outputs * labels / mean(abs(labels))**2), then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdjMSELoss_11518',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'adj_power': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    k_outputs, k_labels = jax.random.split(jax.random.key(0))
    outputs = jax.random.uniform(k_outputs, shape, dtype)
    labels = jax.random.uniform(k_labels, shape, dtype)
    return outputs, labels


def workload(outputs, labels):
    loss = jnp.abs(outputs - labels)
    adj_fact = jnp.mean(jnp.abs(labels)) ** CONFIG['adj_power']
    adj = jnp.exp(-outputs * labels / adj_fact)
    return jnp.mean(loss * adj)
