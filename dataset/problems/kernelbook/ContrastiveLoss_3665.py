"""[bf16 variant] ContrastiveLoss: pairwise_distance (L2 over last dim, eps=1e-6) -> broadcast contrastive loss -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_3665',
    # torch broadcasts the (N,N,N) distance against the (N,N,N,N) label, so all dims must be equal.
    'n': 64,
    'margin': 2.0,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    n = CONFIG['n']
    shape = (n, n, n, n)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    label = jax.random.uniform(k3, shape, dtype)
    return output1, output2, label


def workload(output1, output2, label):
    diff = output1 - output2 + CONFIG['eps']
    d = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # shape (A,B,C), right-aligned broadcast vs label
    hinge = jnp.maximum(CONFIG['margin'] - d, 0.0)
    loss = (1.0 - label) * (d * d) + label * (hinge * hinge)
    return jnp.mean(loss)
