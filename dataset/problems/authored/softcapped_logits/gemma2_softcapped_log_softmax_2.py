"""Gemma-2 final-layer tanh logit soft-capping fused with log-softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcapped_log_softmax',
    'num_tokens': 1024,
    'vocab_size': 262144,
    'softcap': 30.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    logits = jax.random.normal(
        key, (CONFIG['num_tokens'], CONFIG['vocab_size']), dtype=dtype
    ) * 10.0
    return (logits,)


def workload(logits):
    softcap = CONFIG['softcap']
    x = logits.astype(jnp.float32)
    capped = softcap * jnp.tanh(x / softcap)
    return jax.nn.log_softmax(capped, axis=-1)
