"""[bf16 variant] FrmScrLoss: temporal top-k/bottom-k mean gap weighted by label plus categorical action-vs-background abs gap."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FrmScrLoss_9283',
    'n': 128,
    't': 128,
    'c': 512,
    'label_batch': 8,
    'proportion': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    n, t, c = CONFIG['n'], CONFIG['t'], CONFIG['c']
    frm_scrs = jax.random.uniform(k1, (n, t, c), dtype)
    # PyTorch broadcasting requires label shape (L0, n, t, c) with t == n.
    label = jax.random.uniform(k2, (CONFIG['label_batch'], n, t, c), dtype)
    return frm_scrs, label


def workload(frm_scrs, label):
    _n, t, _c = frm_scrs.shape
    k = max(t // CONFIG['proportion'], 1)
    xt = jnp.swapaxes(frm_scrs, 1, 2)  # (n, c, t): top_k works on the last axis
    mean_max = lax.top_k(xt, k)[0].mean(-1)  # (n, c)
    mean_min = -lax.top_k(-xt, k)[0].mean(-1)  # (n, c)
    temporal = ((mean_min - mean_max) * label).sum(-1).mean(-1)  # (L0, L1)

    scaled = frm_scrs * label[:, None]  # (L0, 1, n, t, c)
    act = scaled[..., :-1].max(-1)
    bck = scaled[..., -1]
    categorical = -1.0 * jnp.abs(act - bck).mean(-1).mean(-1)  # (L0, 1)
    return temporal + categorical
