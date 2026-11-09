"""[bf16 variant] Dot-product attention energies: sum(hidden * encoded, axis=2) -> transpose -> softmax(dim=1) -> unsqueeze(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionLayer_10873',
    'seq_len': 512,
    'batch': 256,
    'hidden_size': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    hidden = jax.random.uniform(
        k1, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['hidden_size']), dtype)
    encoded = jax.random.uniform(
        k2, (CONFIG['batch'], CONFIG['hidden_size']), dtype)
    return hidden, encoded


def workload(hidden, encoded):
    # encoded (B, H) broadcasts against hidden (A, B, H) on the trailing axes.
    energies = jnp.sum(hidden * encoded, axis=2)  # (A, B)
    energies = energies.T  # (B, A)
    probs = jax.nn.softmax(energies, axis=1)  # (B, A)
    return probs[:, None, :]  # (B, 1, A)
