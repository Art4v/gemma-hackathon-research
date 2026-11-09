"""[bf16 variant] Multi-head attention with k-attention pooling of K/V plus LayerNorm'd absolute-position attention, dense, residual LayerNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MultiHeadAttentionWithPooling_7846',
    'batch': 16,
    'seq_len': 1024,
    'hidden_size': 1024,
    'n_heads': 4,
    'k_heads': 128,
    'attn_scale_factor': 2,
    'layer_norm_eps': 1.0,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    B, S, H, K = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size'], CONFIG['k_heads']
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(ks[0], (B, S, H), dtype)
    pos = jax.random.uniform(ks[1], (B, S, H), dtype)

    def w(k, shape):
        return jax.random.normal(k, shape, dtype) * 0.02

    zeros = jnp.zeros((H,), dtype)
    ones = jnp.ones((H,), dtype)
    return (
        x, pos,
        w(ks[2], (H, H)), zeros,   # query
        w(ks[3], (H, H)), zeros,   # key
        w(ks[4], (H, H)), zeros,   # value
        w(ks[5], (H, K)),          # theta_k for key pooling
        w(ks[6], (H, K)),          # theta_k for value pooling
        w(ks[7], (H, H)), zeros,   # pos_q_linear
        w(ks[8], (H, H)), zeros,   # pos_k_linear
        ones, zeros,               # pos_ln
        w(ks[9], (H, H)), zeros,   # dense
        ones, zeros,               # LayerNorm
    )


def _linear(x, w, b):
    return jnp.einsum('bsi,io->bso', x, w, precision=_HI) + b


def _layer_norm(x, g, b, eps):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * g + b


def _heads(x, n):
    B, S, H = x.shape
    return x.reshape(B, S, n, H // n).transpose(0, 2, 1, 3)


def _pool(x, theta):
    a = jax.nn.softmax(jnp.einsum('bsh,hk->bsk', x, theta, precision=_HI), axis=1)
    return jnp.einsum('bsh,bsk->bkh', x, a, precision=_HI)


def workload(x, pos_emb, wq, bq, wk, bk, wv, bv, theta_key, theta_value,
             wpq, bpq, wpk, bpk, pos_ln_g, pos_ln_b, wd, bd, ln_g, ln_b):
    n = CONFIG['n_heads']
    eps = CONFIG['layer_norm_eps']
    B, S, H = x.shape
    d = H // n

    q = _heads(_linear(x, wq, bq), n)
    mk = _linear(x, wk, bk)
    mv = _linear(x, wv, bv)
    k = _heads(_pool(mk, theta_key), n)
    v = _heads(_pool(mv, theta_value), n)

    scores = jnp.einsum('bhsd,bhkd->bhsk', q, k, precision=_HI) / (d ** 0.5)
    probs = jax.nn.softmax(scores, axis=-2)
    ctx_item = jnp.einsum('bhsk,bhkd->bhsd', probs, v, precision=_HI)

    v_pos = _heads(mv, n)
    p = _layer_norm(pos_emb, pos_ln_g, pos_ln_b, eps)
    pos_scaling = float(d * CONFIG['attn_scale_factor']) ** -0.5
    pq = _heads(_linear(p, wpq, bpq), n) * pos_scaling
    pk = _heads(_linear(p, wpk, bpk), n)
    bias = jnp.einsum('bhsd,bhtd->bhst', pq, pk, precision=_HI) / (d ** 0.5)
    bias = jax.nn.softmax(bias, axis=-2)
    ctx_pos = jnp.einsum('bhst,bhtd->bhsd', bias, v_pos, precision=_HI)

    ctx = (ctx_item + ctx_pos).transpose(0, 2, 1, 3).reshape(B, S, H)
    out = _linear(ctx, wd, bd)
    return _layer_norm(out + x, ln_g, ln_b, eps)
