"""Model1_9598: squeeze(1) -> (Linear -> ReLU) x3 -> Linear -> LogSoftmax(dim=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Model1_9598',
    'batch': 64,
    'channels': 16,
    'height': 128,
    'input_dim': 512,
    'hidden1': 256,
    'hidden2': 256,
    'hidden3': 256,
    'output_dim': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(
        keys[0], (c['batch'], c['channels'], c['height'], c['input_dim']), dtype)
    dims = [c['input_dim'], c['hidden1'], c['hidden2'], c['hidden3'], c['output_dim']]
    params = []
    for i in range(4):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    # torch squeeze(1) is a no-op unless dim 1 has size 1.
    if x.shape[1] == 1:
        x = jnp.squeeze(x, axis=1)
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    h = h @ w4 + b4
    # LogSoftmax(dim=1) normalizes over axis 1, not the feature axis.
    return jax.nn.log_softmax(h, axis=CONFIG['softmax_axis'])
