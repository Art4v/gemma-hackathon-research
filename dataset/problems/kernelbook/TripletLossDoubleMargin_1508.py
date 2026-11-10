"""[bf16 variant] TripletLossDoubleMargin: squared L2 over dim 1, relu(neg_margin - d_neg) + relu(d_pos - pos_margin), mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TripletLossDoubleMargin_1508',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'pos_margin': 1.0,
    'neg_margin': 3.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    ka, kp, kn = jax.random.split(jax.random.key(0), 3)
    anchor = jax.random.uniform(ka, shape, dtype)
    positive = jax.random.uniform(kp, shape, dtype)
    negative = jax.random.uniform(kn, shape, dtype)
    return anchor, positive, negative


def workload(anchor, positive, negative):
    d_pos = jnp.sum((anchor - positive) ** 2, axis=1)
    d_neg = jnp.sum((anchor - negative) ** 2, axis=1)
    losses = jax.nn.relu(CONFIG['neg_margin'] - d_neg) + jax.nn.relu(d_pos - CONFIG['pos_margin'])
    return jnp.mean(losses)
