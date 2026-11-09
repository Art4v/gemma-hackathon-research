"""[m1 M1_batch variant of authored/norm_residual_fusion/gemma_sandwich_norm_residual] Gemma-2 sandwich norm + residual add, vmapped over a leading pipeline-stage axis where each stage applies its own layer's norm weights to its own microbatch."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma_sandwich_norm_residual_m1_batch',
    'num_stages': 2,
    'batch': 2,  # microbatch per stage
    'seq_len': 4096,
    'hidden': 3584,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_branch, k_res = jax.random.split(key, 2)
    P = CONFIG['num_stages']
    shape = (P, CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden'])
    branch_out = jax.random.normal(k_branch, shape, dtype=dtype)
    residual = jax.random.normal(k_res, shape, dtype=dtype)
    # Gemma stores RMSNorm scale as zero-centred offset: effective scale = 1 + w.
    # Each stage holds a different layer, so the norm weights are stacked per stage.
    post_attn_norm_w = jnp.zeros((P, CONFIG['hidden']), dtype=dtype)
    pre_ffn_norm_w = jnp.zeros((P, CONFIG['hidden']), dtype=dtype)
    return branch_out, residual, post_attn_norm_w, pre_ffn_norm_w


def _gemma_rmsnorm(x, w, eps, out_dtype):
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    y = xf * jax.lax.rsqrt(var + eps)
    y = y * (1.0 + w.astype(jnp.float32))
    return y.astype(out_dtype)


def _stage(branch_out, residual, post_attn_norm_w, pre_ffn_norm_w):
    eps = CONFIG['eps']
    dtype = residual.dtype
    # Post-attention norm on the branch output.
    branch_normed = _gemma_rmsnorm(branch_out, post_attn_norm_w, eps, dtype)
    # Residual add (f32 accumulate, stored in activation dtype).
    new_residual = (residual.astype(jnp.float32)
                    + branch_normed.astype(jnp.float32)).astype(dtype)
    # Pre-feedforward norm on the updated residual stream.
    normed_hidden = _gemma_rmsnorm(new_residual, pre_ffn_norm_w, eps, dtype)
    return normed_hidden, new_residual


def workload(branch_out, residual, post_attn_norm_w, pre_ffn_norm_w):
    return jax.vmap(_stage)(branch_out, residual, post_attn_norm_w, pre_ffn_norm_w)
