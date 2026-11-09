"""ContrastiveLoss: pairwise_distance -> margin hinge -> weighted squared sum -> mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ContrastiveLoss_5598',
    # All four dims must be equal: PyTorch broadcasts the (N,N,N) distance against the
    # (N,N,N,N) is_negative tensor by trailing-dim alignment, as in the original module.
    'n': 64,
    'margin': 2.0,
    'eps': 1e-6,  # F.pairwise_distance default
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['n']
    shape = (n, n, n, n)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    output1 = jax.random.uniform(k1, shape, dtype)
    output2 = jax.random.uniform(k2, shape, dtype)
    is_negative = jax.random.uniform(k3, shape, dtype)
    return output1, output2, is_negative


def workload(output1, output2, is_negative):
    diff = output1 - output2 + CONFIG['eps']
    dist = jnp.sqrt(jnp.sum(diff * diff, axis=-1))  # (N, N, N), keepdim=False
    hinge = jnp.maximum(CONFIG['margin'] - dist, 0.0)
    loss = (1.0 - is_negative) * dist ** 2 + is_negative * hinge ** 2
    return jnp.mean(loss)
