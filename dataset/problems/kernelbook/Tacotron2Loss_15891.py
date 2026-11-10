"""Tacotron2Loss: MSE(mel_before) + MSE(mel_after) + BCEWithLogits(gate), mean-reduced."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Tacotron2Loss_15891',
    'n_outputs': 4,
    'batch': 32,
    'n_mels': 128,
    'frames': 1024,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['n_outputs'], CONFIG['batch'], CONFIG['n_mels'], CONFIG['frames'])
    model_output = jax.random.uniform(k1, shape, dtype)
    targets = jax.random.uniform(k2, shape, dtype)
    return model_output, targets


def workload(model_output, targets):
    mel_target, gate_target = targets[0], targets[1]
    mel_before, mel_after, gate_out = model_output[0], model_output[1], model_output[2]
    mel_loss = jnp.mean((mel_before - mel_target) ** 2) + jnp.mean((mel_after - mel_target) ** 2)
    x = gate_out.reshape(-1)
    y = gate_target.reshape(-1)
    bce = jnp.maximum(x, 0) - x * y + jnp.log1p(jnp.exp(-jnp.abs(x)))
    gate_loss = jnp.mean(bce)
    return mel_loss + gate_loss
