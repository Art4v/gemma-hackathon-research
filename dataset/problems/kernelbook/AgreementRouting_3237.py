"""AgreementRouting: softmax routing logits -> weighted capsule sum -> squash, iterated agreement updates."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AgreementRouting_3237',
    'batch_size': 8,
    'input_caps': 128,
    'output_caps': 128,
    'output_dim': 128,
    'n_iterations': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    u_predict = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['input_caps'], CONFIG['output_caps'], CONFIG['output_dim']),
        dtype,
    )
    b = jnp.zeros((CONFIG['input_caps'], CONFIG['output_caps']), dtype)
    return u_predict, b


def _squash(x):
    lengths2 = jnp.sum(x ** 2, axis=2)
    lengths = jnp.sqrt(lengths2)
    return x * (lengths2 / (1 + lengths2) / lengths)[..., None]


def workload(u_predict, b):
    batch_size, input_caps, output_caps = u_predict.shape[:3]
    c = jax.nn.softmax(b, axis=-1)
    s = jnp.sum(c[..., None] * u_predict, axis=1)
    v = _squash(s)
    if CONFIG['n_iterations'] > 0:
        b_batch = jnp.broadcast_to(b, (batch_size, input_caps, output_caps))
        for _ in range(CONFIG['n_iterations']):
            v = v[:, None, :, :]
            b_batch = b_batch + jnp.sum(u_predict * v, axis=-1)
            c = jax.nn.softmax(b_batch, axis=-1)[..., None]
            s = jnp.sum(c * u_predict, axis=1)
            v = _squash(s)
    return v
