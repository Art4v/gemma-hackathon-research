"""[bf16 variant] WeightedPool: tensordot(x, w) -> additive mask logits (5D broadcast) -> softmax(dim=1) -> matmul(x^T, alphas) -> squeeze(2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'WeightedPool_13858',
    'batch': 8,          # x dim 0
    'seq': 128,          # x dims 1 and 2; must equal dim for the torch broadcasting to be valid
    'dim': 128,          # x last dim, weight rows
    'mask_lead': 8,      # mask dim 0 (becomes output dim 0 via broadcasting)
    'mask_last': 128,    # mask dim 3 (becomes output last dim)
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_m, k_w = jax.random.split(jax.random.key(0), 3)
    b, s, d = CONFIG['batch'], CONFIG['seq'], CONFIG['dim']
    x = jax.random.uniform(k_x, (b, s, s, d), dtype)
    mask = jax.random.uniform(k_m, (CONFIG['mask_lead'], b, s, CONFIG['mask_last']), dtype)
    weight = jax.random.normal(k_w, (d, 1), dtype) * 0.02
    return x, mask, weight


def workload(x, mask, weight):
    alpha = jnp.tensordot(x, weight, axes=1)                       # (B, S, S, 1)
    m = mask.astype(jnp.float32)[:, :, None, :, :]                 # (M0, B, 1, S, M3)
    logits = alpha + (1.0 - m) * CONFIG['mask_value']              # (M0, B, S, S, M3)
    alphas = jax.nn.softmax(logits, axis=1)
    pooled = jnp.matmul(jnp.swapaxes(x, 1, 2), alphas)             # (M0, B, S, S, M3)
    # torch squeeze(2) is a no-op when that dim is not 1; jnp.squeeze would raise.
    if pooled.shape[2] == 1:
        pooled = jnp.squeeze(pooled, axis=2)
    return pooled
