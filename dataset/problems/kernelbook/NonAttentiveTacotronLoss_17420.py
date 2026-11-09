"""[bf16 variant] NonAttentiveTacotronLoss: L1+MSE on prenet/postnet mels vs target, MSE on hop/sample-rate scaled durations."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NonAttentiveTacotronLoss_17420',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'sample_rate': 4,
    'hop_size': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    keys = jax.random.split(jax.random.key(0), 5)
    return tuple(jax.random.uniform(k, shape, dtype) for k in keys)


def workload(prenet_mels, postnet_mels, model_durations, target_durations, target_mels):
    hop = CONFIG['hop_size']
    sr = CONFIG['sample_rate']

    def l1(a, b):
        return jnp.mean(jnp.abs(a - b))

    def mse(a, b):
        return jnp.mean(jnp.square(a - b))

    loss_prenet = l1(prenet_mels, target_mels) + mse(prenet_mels, target_mels)
    loss_postnet = l1(postnet_mels, target_mels) + mse(postnet_mels, target_mels)
    md = model_durations * hop / sr
    td = target_durations * hop / sr
    loss_durations = mse(md, td)
    return loss_prenet, loss_postnet, loss_durations
