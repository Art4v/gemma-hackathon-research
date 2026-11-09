"""[bf16 variant] SoftAttention: tanh(Linear(x)) @ ua -> softmax over sequence -> attention-weighted sum of x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SoftAttention_11380',
    'batch': 32,
    'seq_len': 1024,
    'hidden_dim': 1024,
    'ua_std': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb, ku = jax.random.split(jax.random.key(0), 4)
    n, t, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_dim']
    x = jax.random.uniform(kx, (n, t, h), dtype)
    w = jax.random.normal(kw, (h, h), dtype) * 0.02
    b = jax.random.normal(kb, (h,), dtype) * 0.02
    ua = jax.random.normal(ku, (h,), dtype) * CONFIG['ua_std']
    return x, w, b, ua


def workload(x, w, b, ua):
    hp = lax.Precision.HIGHEST
    va = jnp.tanh(jnp.matmul(x, w, precision=hp) + b)
    scores = jnp.matmul(va, ua, precision=hp)
    omega = jax.nn.softmax(scores, axis=1)
    return jnp.einsum('nth,nt->nh', x, omega, precision=hp)
