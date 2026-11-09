"""[bf16 variant] Word2Vec: Linear -> Linear -> log_softmax over dim 1 on a 4D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Word2Vec_4253',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'features': 512,
    'embedding_size': 256,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    f, e = CONFIG['features'], CONFIG['embedding_size']
    x = jax.random.uniform(
        k[0], (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], f), dtype)
    w1 = jax.random.normal(k[1], (f, e), dtype) * 0.02
    b1 = jax.random.normal(k[2], (e,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (e, f), dtype) * 0.02
    b2 = jax.random.normal(k[4], (f,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = x.astype(w1.dtype) @ w1 + b1
    y = h @ w2 + b2
    return jax.nn.log_softmax(y, axis=CONFIG['softmax_axis'])
