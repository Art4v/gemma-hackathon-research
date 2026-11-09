"""Additive attention: concat(hidden, encoder_outputs) -> Linear -> ReLU -> dot v -> softmax over time."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_8837',
    'timestep': 512,
    'batch': 64,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    t, b, h = CONFIG['timestep'], CONFIG['batch'], CONFIG['hidden_size']
    k = jax.random.split(jax.random.key(0), 5)
    hidden = jax.random.uniform(k[0], (b, h), dtype)
    encoder_outputs = jax.random.uniform(k[1], (t, b, h), dtype)
    w = jax.random.normal(k[2], (2 * h, h), dtype) * 0.02
    bias = jax.random.normal(k[3], (h,), dtype) * 0.02
    v = jax.random.normal(k[4], (h,), dtype) * 0.02
    return hidden, encoder_outputs, w, bias, v


def workload(hidden, encoder_outputs, w, bias, v):
    t = encoder_outputs.shape[0]
    enc = jnp.transpose(encoder_outputs, (1, 0, 2))  # (B, T, H)
    h = jnp.broadcast_to(hidden[:, None, :], (hidden.shape[0], t, hidden.shape[1]))
    x = jnp.concatenate([h, enc], axis=2)  # (B, T, 2H)
    energy = jax.nn.relu(x @ w + bias)  # (B, T, H)
    scores = jnp.einsum('bth,h->bt', energy, v)
    return jax.nn.softmax(scores, axis=1)[:, None, :]  # (B, 1, T)
