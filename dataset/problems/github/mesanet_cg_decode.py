"""[bf16 variant] MesaNet decode step: gated H_kk/H_kv state update, 30-iter CG solve of (H_kk + diag(lamb)) x = q, readout o = x^T H_kv."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mesanet_cg_decode',
    'batch': 128,
    'heads': 16,
    'head_dim': 128,
    'n_hist': 32,
    'max_cg_iters': 30,
    'cg_eps': 1e-5,
    'g_shift': 3.0,
    'lamb_min': 0.25,
}


def _l2norm(x):
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def create_inputs(dtype=jnp.bfloat16):
    B, H, D, N = CONFIG['batch'], CONFIG['heads'], CONFIG['head_dim'], CONFIG['n_hist']
    ks_ = jax.random.split(jax.random.key(0), 8)
    q = jax.random.normal(ks_[0], (B, H, D), dtype)
    k = _l2norm(jax.random.normal(ks_[1], (B, H, D), dtype))
    v = jax.random.normal(ks_[2], (B, H, D), dtype)
    g = jax.nn.log_sigmoid(jax.random.normal(ks_[3], (B, H), dtype) + CONFIG['g_shift'])
    lamb = jax.random.uniform(ks_[4], (H, D), dtype) + CONFIG['lamb_min']
    beta = jax.random.uniform(ks_[5], (B, H), dtype)
    hist_k = _l2norm(jax.random.normal(ks_[6], (B, H, N, D), dtype))
    hist_v = jax.random.normal(ks_[7], (B, H, N, D), dtype)
    hp = lax.Precision.HIGHEST
    prev_h_kk = jnp.einsum('bhnd,bhne->bhde', hist_k, hist_k, precision=hp)
    prev_h_kv = jnp.einsum('bhnd,bhne->bhde', hist_k, hist_v, precision=hp)
    return q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv


def _matvec(x, h):
    # (x[..., None] * h).sum(-2) in the reference; f32-exact contraction.
    return jnp.einsum('bhd,bhde->bhe', x, h, precision=lax.Precision.HIGHEST)


def workload(q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv):
    eps = CONFIG['cg_eps']
    f32 = jnp.float32
    q, k, v, g, lamb, beta = (a.astype(f32) for a in (q, k, v, g, lamb, beta))
    prev_h_kk = prev_h_kk.astype(f32)
    prev_h_kv = prev_h_kv.astype(f32)

    decay = jnp.exp(g)[..., None, None]
    k_beta = k * beta[..., None]
    h_kk = prev_h_kk * decay + k_beta[..., :, None] * k[..., None, :]
    h_kv = prev_h_kv * decay + k_beta[..., :, None] * v[..., None, :]

    diag_h = jnp.diagonal(h_kk, axis1=-2, axis2=-1)
    x = q / (diag_h + lamb)
    r = q - _matvec(x, h_kk) - lamb * x
    p = r
    delta_old = jnp.sum(r * r, axis=-1)

    def body(_, carry):
        x, r, p, delta_old = carry
        ap = _matvec(p, h_kk) + lamb * p
        alpha = delta_old / (jnp.sum(p * ap, axis=-1) + eps)
        x = x + alpha[..., None] * p
        r = r - alpha[..., None] * ap
        delta_new = jnp.sum(r * r, axis=-1)
        cg_beta = delta_new / (delta_old + eps)
        p = r + cg_beta[..., None] * p
        return x, r, p, delta_new

    x, _, _, _ = lax.fori_loop(0, CONFIG['max_cg_iters'], body, (x, r, p, delta_old))
    o = _matvec(x, h_kv)
    return o, h_kk, h_kv
