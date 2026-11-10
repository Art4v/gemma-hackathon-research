"""[bf16 variant] Triplet margin hinge: clamp(margin + repr[:, 1:] - repr[:, :1], 0) -> mean(dim=1) -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TripletMarginLoss_8554',
    'batch': 8192,
    'queue_len_plus': 4096,
    'margin': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    repr_ = jax.random.uniform(key, (CONFIG['batch'], CONFIG['queue_len_plus']), dtype)
    return (repr_,)


def workload(repr_):
    pos = repr_[:, :1]
    neg = repr_[:, 1:]
    dist_hinge = jnp.maximum(CONFIG['margin'] + neg - pos, 0.0)
    return jnp.mean(jnp.mean(dist_hinge, axis=1))
