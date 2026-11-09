"""[bf16 variant] Music Transformer relative attention: QKV projection -> QK^T + skewed relative logits (q @ E^T) -> scale -> softmax -> @V -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'music_transformer_skew_relattn',
    'batch': 8,
    'seq_len': 1024,
    'n_state': 1024,
    'heads': 16,
    'max_sequence': 2048,
}


def create_inputs(dtype=jnp.bfloat16):
    b, s, n = CONFIG['batch'], CONFIG['seq_len'], CONFIG['n_state']
    h, m = CONFIG['heads'], CONFIG['max_sequence']
    kx, ka, kp, ke = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (b, s, n), dtype)
    w_attn = jax.random.normal(ka, (n, 3 * n), dtype) * 0.02
    b_attn = jnp.zeros((3 * n,), dtype)
    w_proj = jax.random.normal(kp, (n, n), dtype) * 0.02
    b_proj = jnp.zeros((n,), dtype)
    e = jax.random.normal(ke, (h, m, n // h), dtype) * 0.02
    return x, w_attn, b_attn, w_proj, b_proj, e


def workload(x, w_attn, b_attn, w_proj, b_proj, e):
    b, s, n = x.shape
    h, m, d = e.shape
    c = x @ w_attn + b_attn
    q, k, v = jnp.split(c, 3, axis=-1)
    q, k, v = (t.reshape(b, s, h, d).transpose(0, 2, 1, 3) for t in (q, k, v))
    w = q @ jnp.swapaxes(k, -1, -2)
    # Skew trick copied exactly from the torch module, including the
    # upper-triangle spill-over values (the original applies no mask).
    q_ = q.transpose(1, 0, 2, 3).reshape(h, b * s, d)
    rel = q_ @ jnp.swapaxes(e[:, m - s:], -1, -2)
    rel = rel.reshape(h, b, s, s)
    rel = jnp.pad(rel, ((0, 0), (0, 0), (0, 0), (1, 0)))
    rel = rel.reshape(h, b, s + 1, s)[:, :, 1:]
    rel = rel.transpose(1, 0, 2, 3)
    w = (w + rel) * (1.0 / d ** 0.5)
    w = jax.nn.softmax(w, axis=-1)
    a = (w @ v).transpose(0, 2, 1, 3).reshape(b, s, n)
    return a @ w_proj + b_proj
