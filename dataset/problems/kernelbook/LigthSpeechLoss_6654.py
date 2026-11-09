"""[bf16 variant] LigthSpeechLoss: MSE(mel, target), L1(cemb_out, cemb), L1(padd_predicted, D) -> three scalar means."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LigthSpeechLoss_6654',
    'shape': (8, 128, 128, 128),
    'num_inputs': 6,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), CONFIG['num_inputs'])
    return tuple(jax.random.uniform(k, CONFIG['shape'], dtype) for k in keys)


def workload(mel, padd_predicted, cemb_out, mel_tac2_target, D, cemb):
    mel_loss = jnp.mean(jnp.square(mel - mel_tac2_target))
    similarity_loss = jnp.mean(jnp.abs(cemb_out - cemb))
    duration_loss = jnp.mean(jnp.abs(padd_predicted - D.astype(jnp.float32)))
    return mel_loss, similarity_loss, duration_loss
