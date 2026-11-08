"""Gemma-2 style final-layer logit tanh soft-capping fused with log-softmax (bf16 in, f32 out)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcapped_log_softmax',
    'batch_size': 4,
    'seq_len': 512,
    'vocab_size': 262144,
    'softcap': 30.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['vocab_size'])
    logits = jax.random.normal(key, shape, dtype=dtype) * 10.0
    return (logits,)


def workload(logits):
    cap = CONFIG['softcap']
    x = logits.astype(jnp.float32)
    capped = cap * jnp.tanh(x / cap)
    return jax.nn.log_softmax(capped, axis=-1)
