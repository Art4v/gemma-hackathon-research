"""[bf16 variant] MultiAccuracy: optional argmax(-1), elementwise equality, sum over last axis, divide by last-dim length."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiAccuracy_12813',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'dim3': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3'])
    k_out, k_tgt = jax.random.split(jax.random.key(0))
    outputs = jax.random.uniform(k_out, shape, dtype)
    targets = jax.random.uniform(k_tgt, shape, dtype)
    return outputs, targets


def workload(outputs, targets):
    # Shapes are static, so this mirrors the PyTorch branch at trace time.
    if outputs.shape != targets.shape:
        outputs = jnp.argmax(outputs, axis=-1)
    matches = jnp.sum(outputs == targets, axis=-1)
    return (matches / targets.shape[-1]).astype(jnp.float32)
