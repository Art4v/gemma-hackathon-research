"""CrossAttentionBlock: LayerNorm -> CLS-query cross-attention -> residual -> LayerNorm -> MLP(GELU) -> residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CrossAttentionBlock_14752',
    'batch': 64,
    'seq_len': 1024,
    'dim': 1024,
    'num_heads': 4,
    'mlp_ratio': 4,
    'eps': 1e-5,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.float32):
    B, N, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    Hd = C * CONFIG['mlp_ratio']
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.uniform(ks[0], (B, N, C), dtype)
    wq = jax.random.normal(ks[1], (C, C), dtype) * 0.02
    wk = jax.random.normal(ks[2], (C, C), dtype) * 0.02
    wv = jax.random.normal(ks[3], (C, C), dtype) * 0.02
    wp = jax.random.normal(ks[4], (C, C), dtype) * 0.02
    bp = jax.random.normal(ks[5], (C,), dtype) * 0.02
    w1 = jax.random.normal(ks[6], (C, Hd), dtype) * 0.02
    b1 = jax.random.normal(ks[7], (Hd,), dtype) * 0.02
    w2 = jax.random.normal(ks[8], (Hd, C), dtype) * 0.02
    b2 = jax.random.normal(ks[9], (C,), dtype) * 0.02
    n1g, n1b = jnp.ones((C,), dtype), jnp.zeros((C,), dtype)
    n2g, n2b = jnp.ones((C,), dtype), jnp.zeros((C,), dtype)
    return (x, n1g, n1b, wq, wk, wv, wp, bp, n2g, n2b, w1, b1, w2, b2)


def _layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * g + b


def workload(x, n1g, n1b, wq, wk, wv, wp, bp, n2g, n2b, w1, b1, w2, b2):
    B, N, C = x.shape
    H = CONFIG['num_heads']
    D = C // H
    h = _layer_norm(x, n1g, n1b)
    q = jnp.matmul(h[:, 0:1], wq, precision=_HI).reshape(B, 1, H, D).transpose(0, 2, 1, 3)
    k = jnp.matmul(h, wk, precision=_HI).reshape(B, N, H, D).transpose(0, 2, 1, 3)
    v = jnp.matmul(h, wv, precision=_HI).reshape(B, N, H, D).transpose(0, 2, 1, 3)
    attn = jnp.matmul(q, jnp.swapaxes(k, -1, -2), precision=_HI) * (D ** -0.5)
    attn = jax.nn.softmax(attn, axis=-1)
    o = jnp.matmul(attn, v, precision=_HI).transpose(0, 2, 1, 3).reshape(B, 1, C)
    o = jnp.matmul(o, wp, precision=_HI) + bp
    x1 = x[:, 0:1] + o
    y = jnp.matmul(_layer_norm(x1, n2g, n2b), w1, precision=_HI) + b1
    y = jax.nn.gelu(y, approximate=False)
    y = jnp.matmul(y, w2, precision=_HI) + b2
    return x1 + y
