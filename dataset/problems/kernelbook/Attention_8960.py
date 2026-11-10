"""[bf16 variant] Attention pooling: matmul score -> squeeze -> softmax over sequence -> weighted sum over sequence."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_8960',
    'batch': 128,
    'seq_len': 1024,
    'attention_size': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['seq_len'], CONFIG['attention_size']),
        dtype,
    )
    attention = jax.random.normal(k_w, (CONFIG['attention_size'], 1), dtype) * 0.02
    return x, attention


def workload(x, attention):
    score = jnp.squeeze(jnp.matmul(x, attention), axis=-1)
    score = jax.nn.softmax(score, axis=1)
    score = jnp.reshape(score, (x.shape[0], x.shape[1], 1))
    scored_x = x * score
    condensed_x = jnp.sum(scored_x, axis=1)
    return condensed_x, score
