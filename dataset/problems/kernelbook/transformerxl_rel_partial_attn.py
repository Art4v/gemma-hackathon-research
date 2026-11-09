"""[bf16 variant] TransformerXL rel-partial attention: qkv/r proj -> AC + rel_shift(BD) -> scale -> softmax -> attn@V -> o_net -> residual -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'transformerxl_rel_partial_attn',
    'qlen': 1024,
    'rlen': 1024,
    'bsz': 8,
    'd_model': 1024,
    'n_head': 16,
    'd_head': 64,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    hd = c['n_head'] * c['d_head']
    ks = jax.random.split(jax.random.key(0), 7)
    w = jax.random.uniform(ks[0], (c['qlen'], c['bsz'], c['d_model']), dtype)
    r = jax.random.uniform(ks[1], (c['rlen'], c['bsz'], c['d_model']), dtype)
    r_w_bias = jax.random.normal(ks[2], (c['n_head'], c['d_head']), dtype) * 0.02
    r_r_bias = jax.random.normal(ks[3], (c['n_head'], c['d_head']), dtype) * 0.02
    w_qkv = jax.random.normal(ks[4], (c['d_model'], 3 * hd), dtype) * 0.02
    w_r = jax.random.normal(ks[5], (c['d_model'], hd), dtype) * 0.02
    w_o = jax.random.normal(ks[6], (hd, c['d_model']), dtype) * 0.02
    ln_g = jnp.ones((c['d_model'],), dtype)
    ln_b = jnp.zeros((c['d_model'],), dtype)
    return w, r, r_w_bias, r_r_bias, w_qkv, w_r, w_o, ln_g, ln_b


def _rel_shift(x):
    # x: (b, n, q, k). Same pad/reshape trick as torch's _rel_shift on (q, k, b, n).
    b, n, q, k = x.shape
    padded = jnp.concatenate([jnp.zeros((b, n, q, 1), x.dtype), x], axis=-1)
    padded = padded.reshape(b, n, k + 1, q)
    return padded[:, :, 1:].reshape(b, n, q, k)


def workload(w, r, r_w_bias, r_r_bias, w_qkv, w_r, w_o, ln_g, ln_b):
    qlen, bsz, _ = w.shape
    rlen = r.shape[0]
    n_head, d_head = r_w_bias.shape[-2:]

    q, k, v = jnp.split(w @ w_qkv, 3, axis=-1)
    klen = k.shape[0]
    q = q.reshape(qlen, bsz, n_head, d_head)
    k = k.reshape(klen, bsz, n_head, d_head)
    v = v.reshape(klen, bsz, n_head, d_head)
    r_k = (r @ w_r).reshape(rlen, bsz, n_head, d_head)

    ac = jnp.einsum('ibnd,jbnd->bnij', q + r_w_bias, k)
    bd = _rel_shift(jnp.einsum('ibnd,jbnd->bnij', q + r_r_bias, r_k))
    score = (ac + bd) * (1.0 / d_head ** 0.5)
    prob = jax.nn.softmax(score, axis=-1)

    vec = jnp.einsum('bnij,jbnd->ibnd', prob, v).reshape(qlen, bsz, n_head * d_head)
    x = w + vec @ w_o

    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * ln_g + ln_b
