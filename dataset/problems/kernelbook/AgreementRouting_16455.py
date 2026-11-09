"""AgreementRouting: softmax(b) -> weighted capsule sum -> squash, refined by n_iterations of routing-by-agreement."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AgreementRouting_16455',
    'batch': 16,
    'input_caps': 2048,
    'output_caps': 128,
    'output_dim': 16,
    'n_iterations': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    u_predict = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['input_caps'], CONFIG['output_caps'], CONFIG['output_dim']),
        dtype,
    )
    # PyTorch initializes the routing logits to zeros.
    b = jnp.zeros((CONFIG['input_caps'], CONFIG['output_caps']), dtype)
    return u_predict, b


def _squash(x):
    lengths2 = jnp.sum(x * x, axis=2)
    lengths = jnp.sqrt(lengths2)
    return x * (lengths2 / (1 + lengths2) / lengths)[:, :, None]


def workload(u_predict, b):
    batch_size, input_caps, output_caps, _ = u_predict.shape
    c = jax.nn.softmax(b, axis=1)
    s = jnp.sum(c[:, :, None] * u_predict, axis=1)
    v = _squash(s)
    b_batch = jnp.broadcast_to(b, (batch_size, input_caps, output_caps))
    for _ in range(CONFIG['n_iterations']):
        b_batch = b_batch + jnp.sum(u_predict * v[:, None], axis=-1)
        c = jax.nn.softmax(b_batch, axis=-1)[..., None]
        s = jnp.sum(c * u_predict, axis=1)
        v = _squash(s)
    return v
