"""[bf16 variant] Self-attention pooling: linear score -> squeeze -> softmax (legacy implicit axis 0 for 4D input) -> weighted sum over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttentionPooling_1698',
    'batch': 8,
    'groups': 128,
    'seq_len': 128,
    'hidden': 256,
    'out_features': 1,
    'softmax_axis': 0,
    'sum_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['groups'], CONFIG['seq_len'], CONFIG['hidden']),
        dtype,
    )
    w = jax.random.normal(k_w, (CONFIG['hidden'], CONFIG['out_features']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_features'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    scores = jnp.squeeze(x @ w + b, axis=-1)
    att_w = jax.nn.softmax(scores, axis=CONFIG['softmax_axis'])
    return jnp.sum(x * att_w[..., None], axis=CONFIG['sum_axis'])
