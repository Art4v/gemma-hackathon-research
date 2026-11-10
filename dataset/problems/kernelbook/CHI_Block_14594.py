"""[bf16 variant] CHI_Block: three LayerNorm + cross-attention residual branches, concat, LayerNorm + MLP(GELU) residual, split."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CHI_Block_14594',
    'batch': 16,
    'seq_len': 1024,
    'dim': 512,
    'num_heads': 4,
    'mlp_hidden_dim': 2048,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    B, N, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    H = CONFIG['mlp_hidden_dim']
    ks = jax.random.split(jax.random.key(0), 12)
    x1 = jax.random.uniform(ks[0], (B, N, C), dtype)
    x2 = jax.random.uniform(ks[1], (B, N, C), dtype)
    x3 = jax.random.uniform(ks[2], (B, N, C), dtype)
    ln_w = jnp.ones((9, C), dtype)
    ln_b = jnp.zeros((9, C), dtype)
    wq = jax.random.normal(ks[3], (3, C, C), dtype) * 0.02
    wk = jax.random.normal(ks[4], (3, C, C), dtype) * 0.02
    wv = jax.random.normal(ks[5], (3, C, C), dtype) * 0.02
    wp = jax.random.normal(ks[6], (3, C, C), dtype) * 0.02
    bp = jax.random.normal(ks[7], (3, C), dtype) * 0.02
    n2_w = jnp.ones((3 * C,), dtype)
    n2_b = jnp.zeros((3 * C,), dtype)
    fc1_w = jax.random.normal(ks[8], (3 * C, H), dtype) * 0.02
    fc1_b = jax.random.normal(ks[9], (H,), dtype) * 0.02
    fc2_w = jax.random.normal(ks[10], (H, 3 * C), dtype) * 0.02
    fc2_b = jax.random.normal(ks[11], (3 * C,), dtype) * 0.02
    return (x1, x2, x3, ln_w, ln_b, wq, wk, wv, wp, bp,
            n2_w, n2_b, fc1_w, fc1_b, fc2_w, fc2_b)


def _ln(x, w, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * w + b


def _attn(xq, xk, xv, wq, wk, wv, wp, bp):
    B, N, C = xq.shape
    h = CONFIG['num_heads']
    d = C // h

    def heads(t):
        return t.reshape(B, N, h, d).transpose(0, 2, 1, 3)

    q, k, v = heads(xq @ wq), heads(xk @ wk), heads(xv @ wv)
    s = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (d ** -0.5)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', p, v).transpose(0, 2, 1, 3).reshape(B, N, C)
    return o @ wp + bp


def workload(x1, x2, x3, ln_w, ln_b, wq, wk, wv, wp, bp,
             n2_w, n2_b, fc1_w, fc1_b, fc2_w, fc2_b):
    def n(i, x):
        return _ln(x, ln_w[i], ln_b[i])

    def a(i, xq, xk, xv):
        return _attn(xq, xk, xv, wq[i], wk[i], wv[i], wp[i], bp[i])

    x1 = x1 + a(0, n(0, x2), n(1, x3), n(2, x1))
    x2 = x2 + a(1, n(3, x1), n(4, x3), n(5, x2))
    x3 = x3 + a(2, n(6, x1), n(7, x2), n(8, x3))
    x = jnp.concatenate([x1, x2, x3], axis=2)
    hdn = jax.nn.gelu(_ln(x, n2_w, n2_b) @ fc1_w + fc1_b, approximate=False)
    x = x + (hdn @ fc2_w + fc2_b)
    C = x.shape[2] // 3
    return x[:, :, :C], x[:, :, C:2 * C], x[:, :, 2 * C:]
