"""TFBCELoss_7164: relu(x) - x*z + log1p(exp(-abs(x))), then sum(-1), then mean(-1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TFBCELoss_7164',
    'batch': 8,
    'd1': 128,
    'd2': 128,
    'd3': 256,
    'pos_weight': 4,  # stored by the PyTorch module, unused in forward
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    key = jax.random.key(0)
    k_logits, k_targets = jax.random.split(key)
    logits = jax.random.uniform(k_logits, shape, dtype)
    targets = jax.random.uniform(k_targets, shape, dtype)
    return logits, targets


def workload(logits, targets):
    relu_logits = jnp.maximum(logits, jnp.zeros_like(logits))
    neg_abs_logits = -jnp.abs(logits)
    term1 = relu_logits - logits * targets
    term2 = jnp.log1p(jnp.exp(neg_abs_logits))
    loss = term1 + term2
    return loss.sum(axis=-1).mean(axis=-1)
