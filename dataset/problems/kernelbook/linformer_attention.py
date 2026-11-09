"""[bf16 variant] Linformer attention: shared low-rank projection of K and V along sequence (Linear E), scaled softmax(Q K_proj), weighted sum with V_proj."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'linformer_attention',
    'batch': 8,
    'heads': 16,
    'input_size': 2048,
    'dim': 128,
    'dim_k': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kq, kk, kv, kw, kb = jax.random.split(jax.random.key(0), 5)
    shape = (c['batch'], c['heads'], c['input_size'], c['dim'])
    q = jax.random.uniform(kq, shape, dtype)
    k = jax.random.uniform(kk, shape, dtype)
    v = jax.random.uniform(kv, shape, dtype)
    # E (shared with F) stored as (input_size, dim_k), i.e. torch weight transposed.
    w = jax.random.normal(kw, (c['input_size'], c['dim_k']), dtype) * 0.02
    b = jax.random.normal(kb, (c['dim_k'],), dtype) * 0.02
    return q, k, v, w, b


def workload(q, k, v, w, b):
    dim_k = w.shape[1]
    kp = jnp.einsum('bhnd,nk->bhdk', k, w) + b
    scores = jnp.einsum('bhnd,bhdk->bhnk', q, kp) / jnp.sqrt(jnp.asarray(dim_k, q.dtype))
    p = jax.nn.softmax(scores, axis=-1)
    vp = jnp.einsum('bhnd,nk->bhdk', v, w) + b
    return jnp.einsum('bhnk,bhdk->bhnd', p, vp)
