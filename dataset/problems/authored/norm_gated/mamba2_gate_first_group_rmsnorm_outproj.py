"""Mamba2 gated group RMSNorm (gate-before-norm: y*silu(z) -> per-group RMSNorm -> weight) fused with out_proj matmul."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mamba2_gate_first_group_rmsnorm_outproj',
    'batch': 4,
    'seq_len': 4096,
    'd_inner': 5120,
    'ngroups': 8,
    'd_model': 2560,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_y, k_z, k_w = jax.random.split(key, 3)
    B, L, D, M = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_inner'], CONFIG['d_model']
    y = jax.random.normal(k_y, (B, L, D), dtype)
    z = jax.random.normal(k_z, (B, L, D), dtype)
    norm_weight = jnp.ones((D,), dtype)
    w_out = jax.random.normal(k_w, (D, M), dtype) * 0.02
    return y, z, norm_weight, w_out


def workload(y, z, norm_weight, w_out):
    out_dtype = y.dtype
    G = CONFIG['ngroups']
    eps = CONFIG['eps']
    B, L, D = y.shape
    gs = D // G
    # Gate first (norm_before_gate=False), in float32.
    zf = z.astype(jnp.float32)
    g = y.astype(jnp.float32) * (zf * jax.nn.sigmoid(zf))
    # Per-group RMS normalization.
    g = g.reshape(B, L, G, gs)
    var = jnp.mean(g * g, axis=-1, keepdims=True)
    g = g * jax.lax.rsqrt(var + eps)
    g = g.reshape(B, L, D) * norm_weight.astype(jnp.float32)
    h = g.astype(out_dtype)
    # Output projection with float32 accumulation.
    out = jnp.einsum('bld,dm->blm', h, w_out, preferred_element_type=jnp.float32)
    return out.astype(out_dtype)
