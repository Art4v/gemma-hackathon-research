"""[bf16 variant] LinearWithChannel: x repeated over channels, batched matmul with per-channel weight, plus per-channel bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearWithChannel_10279',
    'batch_size': 4096,
    'input_size': 1024,
    'output_size': 1024,
    'channel_size': 32,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    i = CONFIG['input_size']
    o = CONFIG['output_size']
    c = CONFIG['channel_size']
    x = jax.random.uniform(k_x, (b, i), dtype)
    weight = jax.random.normal(k_w, (c, i, o), dtype) * 0.02
    bias = jax.random.normal(k_b, (c, 1, o), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    # (B, I) @ (C, I, O) broadcasts x over channels -> (C, B, O), same as repeat + bmm.
    return jnp.matmul(x, weight) + bias
