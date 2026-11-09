"""[bf16 variant] Self-attention (fastai): 1x1 conv q/k/v -> f^T g -> softmax(dim=1) -> gamma * h @ beta + x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_2808',
    'batch': 8,
    'channels': 512,
    'height': 64,
    'width': 64,
    'div': 4,
    'gamma': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    cq = c // CONFIG['div'] if c > 1 else c
    kx, kq, kk, kv = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (b, c, h, w), dtype)
    wq = jax.random.normal(kq, (c, cq), dtype) * 0.02
    bq = jnp.zeros((cq,), dtype)
    wk = jax.random.normal(kk, (c, cq), dtype) * 0.02
    bk = jnp.zeros((cq,), dtype)
    wv = jax.random.normal(kv, (c, c), dtype) * 0.02
    bv = jnp.zeros((c,), dtype)
    # ponytail: torch inits gamma=0 (output == x); nonzero here so attention matters.
    gamma = jnp.full((1,), CONFIG['gamma'], dtype)
    return x, wq, bq, wk, bk, wv, bv, gamma


def workload(x, wq, bq, wk, bk, wv, bv, gamma):
    size = x.shape
    xf = x.reshape(size[0], size[1], -1)  # (B, C, N)
    f = jnp.einsum('bcn,co->bon', xf, wq) + bq[None, :, None]
    g = jnp.einsum('bcn,co->bon', xf, wk) + bk[None, :, None]
    h = jnp.einsum('bcn,co->bon', xf, wv) + bv[None, :, None]
    s = jnp.einsum('bon,bom->bnm', f, g)  # (B, N, N) = f^T @ g
    beta = jax.nn.softmax(s, axis=1)
    o = gamma[0] * jnp.einsum('bcn,bnm->bcm', h, beta) + xf
    return o.reshape(size)
