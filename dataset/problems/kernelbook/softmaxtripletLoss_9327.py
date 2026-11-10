"""[bf16 variant] Softmax triplet loss: pairwise L2 distances -> exp -> softmax-ratio -> sum of squares, plus correct-ordering count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'softmaxtripletLoss_9327',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'feature_dim': 128,
    'dist_axis': -1,
    'num_inputs': 3,
    'seed': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['feature_dim'])
    keys = jax.random.split(jax.random.key(CONFIG['seed']), CONFIG['num_inputs'])
    anchor = jax.random.uniform(keys[0], shape, dtype)
    pos = jax.random.uniform(keys[1], shape, dtype)
    neg = jax.random.uniform(keys[2], shape, dtype)
    return anchor, pos, neg


def _dist(a, b):
    d = jnp.square(a - b)
    d = jnp.maximum(d, jnp.zeros_like(d))
    return jnp.sqrt(jnp.sum(d, axis=CONFIG['dist_axis']))


def workload(anchor, pos, neg):
    d2pos = _dist(anchor, pos)
    d2neg = _dist(anchor, neg)
    e_pos = jnp.exp(d2pos)
    e_neg = jnp.exp(d2neg)
    d_pos = e_pos / (e_pos + e_neg)
    loss = jnp.sum(jnp.square(d_pos))
    return loss, jnp.sum(d2pos < d2neg)
