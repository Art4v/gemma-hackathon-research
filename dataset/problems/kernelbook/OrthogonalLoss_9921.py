"""[bf16 variant] OrthogonalLoss: L2-normalize(features, dim=1) -> matmul(descriptor.T) -> mask label-unequal pairs -> mean^2 + clamp(mean sq - 1/dim, 0)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OrthogonalLoss_9921',
    'n': 8192,
    'dim': 2048,
    'num_classes': 128,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    features = jax.random.uniform(k1, (CONFIG['n'], CONFIG['dim']), dtype)
    descriptor = jax.random.uniform(k2, (CONFIG['n'], CONFIG['dim']), dtype)
    labels = jax.random.randint(k3, (CONFIG['n'],), 0, CONFIG['num_classes'], jnp.int32)
    return features, descriptor, labels


def workload(features, descriptor, labels):
    norm = jnp.sqrt(jnp.sum(features * features, axis=1, keepdims=True))
    features = features / jnp.maximum(norm, CONFIG['eps'])
    labels_not_equal = jnp.expand_dims(labels, 1) != jnp.expand_dims(labels, 0)
    # .T reverses all axes, matching torch's .T on N-d tensors.
    sim = jnp.matmul(features, descriptor.T, precision=lax.Precision.HIGHEST)
    neg_dis = sim * labels_not_equal.astype(sim.dtype)
    dim = features.shape[1]
    return jnp.mean(neg_dis) ** 2 + jnp.maximum(jnp.mean(neg_dis ** 2) - 1.0 / dim, 0.0)
