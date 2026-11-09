"""CoPredictor: MLP(GELU) x2 -> Biaffine, plus MLP(GELU) -> Linear on the pair grid, summed."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CoPredictor_15291',
    'batch': 8,
    'seq_len': 256,
    'hid_size': 768,
    'biaffine_size': 256,
    'channels': 128,
    'ffnn_hid_size': 256,
    'cls_num': 8,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    b, l = c['batch'], c['seq_len']
    hid, bs, ch = c['hid_size'], c['biaffine_size'], c['channels']
    ff, cls = c['ffnn_hid_size'], c['cls_num']
    ks = jax.random.split(jax.random.key(0), 12)
    x = jax.random.uniform(ks[0], (b, l, hid), dtype)
    y = jax.random.uniform(ks[1], (b, l, hid), dtype)
    z = jax.random.uniform(ks[2], (b, l, l, ch), dtype)
    w1 = jax.random.normal(ks[3], (hid, bs), dtype) * 0.02
    b1 = jax.random.normal(ks[4], (bs,), dtype) * 0.02
    w2 = jax.random.normal(ks[5], (hid, bs), dtype) * 0.02
    b2 = jax.random.normal(ks[6], (bs,), dtype) * 0.02
    wb = jax.random.normal(ks[7], (cls, bs + 1, bs + 1), dtype) * 0.02
    w3 = jax.random.normal(ks[8], (ch, ff), dtype) * 0.02
    b3 = jax.random.normal(ks[9], (ff,), dtype) * 0.02
    w4 = jax.random.normal(ks[10], (ff, cls), dtype) * 0.02
    b4 = jax.random.normal(ks[11], (cls,), dtype) * 0.02
    return x, y, z, w1, b1, w2, b2, wb, w3, b3, w4, b4


def _mlp(x, w, b):
    return jax.nn.gelu(x @ w + b, approximate=False)


def _append_ones(x):
    return jnp.concatenate([x, jnp.ones_like(x[..., :1])], axis=-1)


def workload(x, y, z, w1, b1, w2, b2, wb, w3, b3, w4, b4):
    h = _append_ones(_mlp(x, w1, b1))
    t = _append_ones(_mlp(y, w2, b2))
    o1 = jnp.einsum('bxi,oij,byj->bxyo', h, wb, t)
    o2 = _mlp(z, w3, b3) @ w4 + b4
    return o1 + o2
