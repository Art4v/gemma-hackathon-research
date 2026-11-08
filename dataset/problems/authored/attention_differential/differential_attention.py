"""Differential attention: dual causal softmax maps (A1 - lambda*A2) @ V with learned re-parameterised lambda, per-head RMSNorm, (1 - lambda_init) scaling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'differential_attention',
    'batch': 4,
    'num_heads': 16,          # differential heads; q/k carry 2 * num_heads sub-heads
    'seq_len': 4096,
    'head_dim': 64,           # per q/k sub-head; V width per head is 2 * head_dim
    'layer_idx': 12,          # depth used for lambda_init schedule
    'lambda_param_std': 0.1,  # init std of lambda re-parameterisation vectors (paper)
    'rms_eps': 1e-5,
    'q_block': 256,           # query block size used by the reference to bound memory
    'causal': True,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    keys = jax.random.split(jax.random.key(0), 7)
    q = jax.random.normal(keys[0], (B, 2 * H, S, D), dtype)
    k = jax.random.normal(keys[1], (B, 2 * H, S, D), dtype)
    v = jax.random.normal(keys[2], (B, H, S, 2 * D), dtype)
    std = CONFIG['lambda_param_std']
    lq1 = jax.random.normal(keys[3], (D,), jnp.float32) * std
    lk1 = jax.random.normal(keys[4], (D,), jnp.float32) * std
    lq2 = jax.random.normal(keys[5], (D,), jnp.float32) * std
    lk2 = jax.random.normal(keys[6], (D,), jnp.float32) * std
    subln_w = jnp.ones((2 * D,), jnp.float32)
    return q, k, v, lq1, lk1, lq2, lk2, subln_w


def workload(q, k, v, lq1, lk1, lq2, lk2, subln_w):
    B, H2, S, D = q.shape
    H = H2 // 2
    Dv = v.shape[-1]
    C = min(CONFIG['q_block'], S)
    nb = S // C
    eps = CONFIG['rms_eps']
    out_dtype = q.dtype

    layer = CONFIG['layer_idx']
    lambda_init = 0.8 - 0.6 * jnp.exp(-0.3 * (layer - 1.0))
    lam = (jnp.exp(jnp.sum(lq1.astype(jnp.float32) * lk1.astype(jnp.float32)))
           - jnp.exp(jnp.sum(lq2.astype(jnp.float32) * lk2.astype(jnp.float32)))
           + lambda_init)
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    vf = v.astype(jnp.float32)
    w = subln_w.astype(jnp.float32)
    k_pos = jnp.arange(S)

    # (nb, B, 2H, C, D)
    qb = q.reshape(B, H2, nb, C, D).transpose(2, 0, 1, 3, 4)

    def block(args):
        qi, bi = args
        s = jnp.einsum('bhqd,bhkd->bhqk', qi, k,
                       preferred_element_type=jnp.float32) * scale
        if CONFIG['causal']:
            q_pos = bi * C + jnp.arange(C)
            mask = q_pos[:, None] >= k_pos[None, :]
            s = jnp.where(mask[None, None], s, -jnp.inf)
        p = jax.nn.softmax(s, axis=-1)                 # (B, 2H, C, S)
        p = p.reshape(B, H, 2, C, S)
        a = p[:, :, 0] - lam * p[:, :, 1]              # (B, H, C, S)
        o = jnp.einsum('bhqk,bhkd->bhqd', a, vf)       # (B, H, C, Dv) f32
        ms = jnp.mean(o * o, axis=-1, keepdims=True)
        o = o * lax.rsqrt(ms + eps) * w
        o = o * (1.0 - lambda_init)
        return o.astype(out_dtype)

    ob = lax.map(block, (qb, jnp.arange(nb)))          # (nb, B, H, C, Dv)
    o = ob.transpose(1, 0, 3, 2, 4).reshape(B, S, H * Dv)
    return o
