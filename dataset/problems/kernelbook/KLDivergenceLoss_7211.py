"""[bf16 variant] KL divergence regularization loss for evidential classification: alpha_tilde -> lgamma + digamma terms -> mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'KLDivergenceLoss_7211',
    'batch': 8,
    'dim_1': 128,
    'dim_2': 128,
    'n_classes': 128,
    'evidence_offset': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim_1'], CONFIG['dim_2'], CONFIG['n_classes'])
    k_evidence, k_target = jax.random.split(jax.random.key(0))
    evidence = jax.random.uniform(k_evidence, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return evidence, target


def workload(evidence, target):
    alpha = evidence + CONFIG['evidence_offset']
    n_classes = evidence.shape[-1]
    alpha_tilde = target + (1 - target) * alpha
    strength_tilde = alpha_tilde.sum(axis=-1)
    first = (
        lax.lgamma(strength_tilde)
        - lax.lgamma(jnp.asarray(float(n_classes), dtype=alpha_tilde.dtype))
        - lax.lgamma(alpha_tilde).sum(axis=-1)
    )
    # Matches the PyTorch source: [:, None] inserts the new axis at position 1, not at the end.
    second = (
        (alpha_tilde - 1) * (lax.digamma(alpha_tilde) - lax.digamma(strength_tilde)[:, None])
    ).sum(axis=-1)
    loss = first + second
    return loss.mean()
