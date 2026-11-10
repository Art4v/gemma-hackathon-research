"""[bf16 variant] Concat(loffset[None], z) -> Linear -> ReLU -> Linear -> Softmax over the concat axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NLKDifferenceOffset_17364',
    'num_z': 8,
    'extra_dims': (8, 8, 8, 128),
    'dim': 128,
    'hidden_dim': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 6)
    dim = CONFIG['dim']
    hidden = CONFIG['hidden_dim']
    scale = CONFIG['weight_scale']
    loffset_shape = CONFIG['extra_dims'] + (dim,)
    z_shape = (CONFIG['num_z'],) + loffset_shape
    loffset = jax.random.uniform(keys[0], loffset_shape, dtype)
    z = jax.random.uniform(keys[1], z_shape, dtype)
    w1 = jax.random.normal(keys[2], (dim, hidden), dtype) * scale
    b1 = jax.random.normal(keys[3], (hidden,), dtype) * scale
    w2 = jax.random.normal(keys[4], (hidden, dim), dtype) * scale
    b2 = jax.random.normal(keys[5], (dim,), dtype) * scale
    return loffset, z, w1, b1, w2, b2


def workload(loffset, z, w1, b1, w2, b2):
    all_emb = jnp.concatenate((jnp.expand_dims(loffset, axis=0), z), axis=0)
    hidden = jnp.maximum(all_emb @ w1 + b1, 0)
    logits = hidden @ w2 + b2
    return jax.nn.softmax(logits, axis=0)
