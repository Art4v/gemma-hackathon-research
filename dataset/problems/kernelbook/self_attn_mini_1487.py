"""self_attn_mini: QKV linear projections -> q @ k.transpose(1,2) / D -> softmax(dim=2) -> @ v on a 4D input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'self_attn_mini_1487',
    'batch': 8,
    'dim1': 256,
    'dim2': 256,
    'input_size': 256,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.float32):
    kx, kq, kk, kv = jax.random.split(jax.random.key(0), 4)
    d = CONFIG['input_size']
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], d), dtype)
    w_q = jax.random.normal(kq, (d, d), dtype) * 0.02
    w_k = jax.random.normal(kk, (d, d), dtype) * 0.02
    w_v = jax.random.normal(kv, (d, d), dtype) * 0.02
    return x, w_q, w_k, w_v


def workload(x, w_q, w_k, w_v):
    d = x.shape[-1]
    q = jnp.matmul(x, w_q, precision=_HI)
    k = jnp.matmul(x, w_k, precision=_HI)
    v = jnp.matmul(x, w_v, precision=_HI)
    # torch.transpose(k, 1, 2) on 4D swaps dims 1,2, so the contraction pairs q's last dim with k's dim 1.
    wt = jnp.einsum('axmj,ajxn->axmn', q, k, precision=_HI) / d
    sm = jax.nn.softmax(wt, axis=2)
    return jnp.einsum('axmn,axnd->axmd', sm, v, precision=_HI)
