"""[bf16 variant] CatRepr: concat(first, mean over seq, last) along last dim, then flatten per item."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CatRepr_9939',
    'num_items': 16,
    'seq': 32,
    'rows': 256,
    'cols': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['num_items'], CONFIG['seq'], CONFIG['rows'], CONFIG['cols']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    # Iterating the torch list over dim 0 is the same as batching over axis 0 here.
    cat = jnp.concatenate([x[:, 0], jnp.mean(x, axis=1), x[:, -1]], axis=-1)
    return cat.reshape(x.shape[0], -1)
