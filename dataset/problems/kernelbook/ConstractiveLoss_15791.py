"""[bf16 variant] Contrastive loss: L2 pairwise_distance over the last dim, then sum((1-y)*d^2 + y*clamp(margin-d,0)^2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConstractiveLoss_15791',
    'dim0': 64,
    'dim1': 64,
    'dim2': 64,
    'dim3': 64,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['dim0'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    out_vec_t0 = jax.random.uniform(k0, shape, dtype)
    out_vec_t1 = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return out_vec_t0, out_vec_t1, label


def workload(out_vec_t0, out_vec_t1, label):
    # F.pairwise_distance(p=2): norm over the last dim of (x1 - x2 + eps).
    diff = out_vec_t0 - out_vec_t1 + CONFIG['eps']
    distance = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
    # distance (A,B,C) broadcasts against label (A,B,C,D) as (1,A,B,C), same as PyTorch.
    hinge = jnp.maximum(CONFIG['margin'] - distance, 0.0)
    loss = (1.0 - label) * distance ** 2 + label * hinge ** 2
    return jnp.sum(loss)
