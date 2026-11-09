"""[bf16 variant] FocalLoss: softmax(last axis) attention * -log_softmax(axis 1) CE, gathered at x[i, target[i]], then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FocalLoss_13061',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'alpha': 1.0,
    'gamma': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(k1, shape, dtype)
    target = jax.random.randint(k2, (CONFIG['batch'],), 0, CONFIG['channels'], dtype=jnp.int32)
    return x, target


def workload(x, target):
    rows = jnp.arange(x.shape[0])
    # PyTorch softmaxes the full tensor over the last axis, then gathers; softmax of the gathered slice is identical.
    prob = jax.nn.softmax(x[rows, target], axis=-1)
    attn = jnp.power(1.0 - prob, CONFIG['gamma'])
    ce = -jax.nn.log_softmax(x, axis=1)[rows, target]
    return jnp.mean(CONFIG['alpha'] * ce * attn)
