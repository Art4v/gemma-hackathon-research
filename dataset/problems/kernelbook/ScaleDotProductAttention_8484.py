"""Scaled dot-product attention with torch 4-D transpose(1,2) semantics: (q @ k^T(1,2)) * scale -> softmax -> @ v."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ScaleDotProductAttention_8484',
    'batch': 8,
    'n': 128,  # torch's 4-D matmul with transpose(1, 2) forces dims 1, 2, 3 to be equal
    'k_dim': 128,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['n'], CONFIG['n'], CONFIG['n'])
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.uniform(kq, shape, dtype)
    k = jax.random.uniform(kk, shape, dtype)
    v = jax.random.uniform(kv, shape, dtype)
    return q, k, v


def workload(q, k, v):
    # k_dim equals k.shape[-1] in the original module (k_dim=4 on (4,4,4,4) inputs).
    scale = 1.0 / (k.shape[-1] ** 0.5)
    # torch.matmul(q, k.transpose(1, 2)) on 4-D tensors: batch dims (b, i), contract q's last with k's axis 1.
    score = jnp.einsum('bijm,bmil->bijl', q, k, precision=lax.Precision.HIGHEST) * scale
    weights = jax.nn.softmax(score, axis=-1)
    return jnp.einsum('bijl,bild->bijd', weights, v, precision=lax.Precision.HIGHEST)
