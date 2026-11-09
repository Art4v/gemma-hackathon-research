"""Attn_14288: sum(hidden * encoder_output, -1) -> transpose -> softmax(dim=1) -> unsqueeze(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attn_14288',
    'seq_len': 1024,
    'batch': 128,
    'hidden_size': 512,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    hidden = jax.random.uniform(
        k1, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['hidden_size']), dtype)
    encoder_output = jax.random.uniform(
        k2, (CONFIG['batch'], CONFIG['hidden_size']), dtype)
    return hidden, encoder_output


def workload(hidden, encoder_output):
    # encoder_output (B, H) broadcasts over hidden's leading axis, as in torch.
    energies = jnp.sum(hidden * encoder_output, axis=2)  # (S, B)
    probs = jax.nn.softmax(energies.T, axis=1)  # (B, S)
    return probs[:, None, :]  # (B, 1, S)
