"""[bf16 variant] SimCLR loss: cosine similarity matrix / T -> symmetric cross-entropy against diagonal labels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimCLRLoss_4832',
    'batch_size': 8192,
    'feature_dim': 2048,
    'temperature': 4.0,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch_size'], CONFIG['feature_dim'])
    f1 = jax.random.uniform(k1, shape, dtype)
    f2 = jax.random.uniform(k2, shape, dtype)
    return f1, f2


def workload(f1, f2):
    eps = CONFIG['eps']
    # Matches torch.cosine_similarity: x.y / sqrt(max(|x|^2 |y|^2, eps^2)).
    dots = f1 @ f2.T
    n1 = jnp.sum(f1 * f1, axis=-1)
    n2 = jnp.sum(f2 * f2, axis=-1)
    denom = jnp.sqrt(jnp.maximum(n1[:, None] * n2[None, :], eps * eps))
    sim = dots / denom / CONFIG['temperature']
    diag = jnp.diagonal(sim)
    loss_rows = jnp.mean(jax.nn.logsumexp(sim, axis=1) - diag)
    loss_cols = jnp.mean(jax.nn.logsumexp(sim, axis=0) - diag)
    return (loss_rows + loss_cols) * 0.5
