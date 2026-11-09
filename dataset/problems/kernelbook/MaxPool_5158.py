"""MaxPool_5158: transpose(1,2) -> max_pool1d over full sequence length -> squeeze, i.e. max over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaxPool_5158',
    'batch': 64,
    'seq_len': 1024,
    'channels': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['seq_len'], CONFIG['channels']), dtype=dtype
    )
    return (x,)


def workload(x):
    # kernel_size == seq_len, stride defaults to kernel_size, no padding: one window over the whole sequence.
    return jnp.max(x, axis=1)
