"""[bf16 variant] SelfAttention_9168: Linear -> tanh -> Linear -> squeeze(2) -> softmax(-1) -> broadcast mul -> sum(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_9168',
    'batch': 8,
    'seq': 8,  # must equal batch: torch's 5-D broadcast lines weights' seq axis up with x's batch axis
    'tokens': 512,
    'input_dim': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    b, s, t, d = CONFIG['batch'], CONFIG['seq'], CONFIG['tokens'], CONFIG['input_dim']
    x = jax.random.uniform(kx, (b, s, t, d), dtype)
    w1 = jax.random.normal(k1, (d, d), dtype) * 0.02
    b1 = jax.random.normal(k2, (d,), dtype) * 0.02
    w2 = jax.random.normal(k3, (d, 1), dtype) * 0.02
    b2 = jax.random.normal(k4, (1,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.tanh(x @ w1 + b1)
    logits = h @ w2 + b2  # (B, S, T, 1)
    # torch squeeze(dim=2) is a no-op unless axis 2 has size 1
    if logits.shape[2] == 1:
        logits = jnp.squeeze(logits, axis=2)
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(x * jnp.expand_dims(weights, 2), axis=1)
