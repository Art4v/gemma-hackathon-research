"""[bf16 variant] MLPAttention: tanh(ctx@Wc + hid@Wh) @ w_mlp / temp -> softmax(dim 0) -> weighted sum of ctx over dim 0 -> @ W_ctx2hid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLPAttention_9503',
    'S': 32,
    'A': 16,
    'B': 64,
    'ctx_dim': 512,
    'hid_dim': 512,
    'temperature': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    S, A, B = CONFIG['S'], CONFIG['A'], CONFIG['B']
    C, H = CONFIG['ctx_dim'], CONFIG['hid_dim']
    mid = C  # att_bottleneck='ctx'
    k = jax.random.split(jax.random.key(0), 6)
    hid = jax.random.uniform(k[0], (S, A, B, H), dtype)
    ctx = jax.random.uniform(k[1], (S, A, B, C), dtype)
    w_ctx2ctx = jax.random.normal(k[2], (C, mid), dtype) * 0.02
    w_hid2ctx = jax.random.normal(k[3], (H, mid), dtype) * 0.02
    w_mlp = jax.random.normal(k[4], (mid, 1), dtype) * 0.02
    w_ctx2hid = jax.random.normal(k[5], (C, H), dtype) * 0.02
    return hid, ctx, w_ctx2ctx, w_hid2ctx, w_mlp, w_ctx2hid


def workload(hid, ctx, w_ctx2ctx, w_hid2ctx, w_mlp, w_ctx2hid):
    inner = ctx @ w_ctx2ctx + hid @ w_hid2ctx
    scores = (jnp.tanh(inner) @ w_mlp)[..., 0] / CONFIG['temperature']
    alpha = jax.nn.softmax(scores, axis=0)
    z = jnp.sum(alpha[..., None] * ctx, axis=0) @ w_ctx2hid
    return alpha, z
