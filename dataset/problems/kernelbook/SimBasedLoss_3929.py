"""SimBasedLoss: L2-normalize(dim=1) -> x @ x.T (full axis reversal, batched) for student and teacher -> MSE."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SimBasedLoss_3929',
    # torch .T on 4D reverses all axes; matmul (B,C,H,W)@(W,H,C,B) needs B == C == H == W.
    'n': 64,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['n']
    k1, k2 = jax.random.split(jax.random.key(0))
    y_s = jax.random.uniform(k1, (n, n, n, n), dtype)
    y_t = jax.random.uniform(k2, (n, n, n, n), dtype)
    return y_s, y_t


def _normalize(x):
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps'])


def _self_sims(x):
    x = _normalize(x)
    return jnp.matmul(x, jnp.transpose(x), precision=lax.Precision.HIGHEST)


def workload(y_s, y_t):
    diff = _self_sims(y_s) - _self_sims(y_t)
    return jnp.mean(diff * diff)
