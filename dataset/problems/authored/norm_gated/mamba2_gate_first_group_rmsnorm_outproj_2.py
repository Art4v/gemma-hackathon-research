"""Mamba2 gate-first grouped RMSNorm: rmsnorm_group(x * silu(z)) * w, followed by out_proj matmul."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mamba2_gate_first_group_rmsnorm_outproj',
    'batch': 4,
    'seq_len': 2048,
    'd_inner': 8192,
    'd_model': 4096,
    'n_groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kz, kw = jax.random.split(key, 3)
    B, L = CONFIG['batch'], CONFIG['seq_len']
    D, M = CONFIG['d_inner'], CONFIG['d_model']
    x = jax.random.normal(kx, (B, L, D), dtype)
    z = jax.random.normal(kz, (B, L, D), dtype)
    norm_weight = jnp.ones((D,), dtype)
    w_out = jax.random.normal(kw, (D, M), dtype) * 0.02
    return x, z, norm_weight, w_out


def workload(x, z, norm_weight, w_out):
    B, L, D = x.shape
    G = CONFIG['n_groups']
    eps = CONFIG['eps']
    xf = x.astype(jnp.float32)
    zf = z.astype(jnp.float32)
    # Gate before normalization (norm_before_gate=False).
    h = xf * (zf * jax.nn.sigmoid(zf))
    hg = h.reshape(B, L, G, D // G)
    var = jnp.mean(hg * hg, axis=-1, keepdims=True)
    hg = hg * jax.lax.rsqrt(var + eps)
    h = hg.reshape(B, L, D) * norm_weight.astype(jnp.float32)
    h = h.astype(x.dtype)
    out = jnp.einsum('bld,dm->blm', h, w_out,
                     preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
