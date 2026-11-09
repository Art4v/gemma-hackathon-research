"""[bf16 variant] OutputLayer: concat -> weight-vector matmul (start/end) -> mask_logits (target - 1e30 * (1 - mask))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'OutputLayer_2119',
    'batch': 128,
    'hidden_size': 128,  # must equal batch: torch broadcasts (B, L) logits against (B, H, L) mask
    'seq_len': 1024,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    b, h, l = CONFIG['batch'], CONFIG['hidden_size'], CONFIG['seq_len']
    k1, k2, k3, k4, k5, k6 = jax.random.split(jax.random.key(0), 6)
    m1 = jax.random.uniform(k1, (b, h, l), dtype)
    m2 = jax.random.uniform(k2, (b, h, l), dtype)
    m3 = jax.random.uniform(k3, (b, h, l), dtype)
    cmask = jax.random.uniform(k4, (b, h, l), dtype)
    w1 = jax.random.normal(k5, (2 * h,), dtype) * 0.02
    w2 = jax.random.normal(k6, (2 * h,), dtype) * 0.02
    return m1, m2, m3, cmask, w1, w2


def _mask_logits(target, mask):
    return target + CONFIG['mask_value'] * (1 - mask.astype(jnp.float32))


def workload(m1, m2, m3, cmask, w1, w2):
    start = jnp.concatenate((m1, m2), axis=1)
    end = jnp.concatenate((m1, m3), axis=1)
    # torch.matmul(1-D (2H,), 3-D (B, 2H, L)) -> (B, L)
    start = jnp.einsum('c,bcl->bl', w1, start, precision=lax.Precision.HIGHEST)
    end = jnp.einsum('c,bcl->bl', w2, end, precision=lax.Precision.HIGHEST)
    # (B, L) + (B, H, L) broadcasts like torch: logits align with mask's (H, L) axes.
    return _mask_logits(start, cmask), _mask_logits(end, cmask)
