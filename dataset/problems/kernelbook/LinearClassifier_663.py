"""LinearClassifier: max over sequence -> Linear -> LeakyReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LinearClassifier_663',
    'batch': 64,
    'seq_length': 512,
    'd_features': 1024,
    'd_hid': 2048,
    'd_out': 1024,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    b, l, f = CONFIG['batch'], CONFIG['seq_length'], CONFIG['d_features']
    h, o = CONFIG['d_hid'], CONFIG['d_out']
    x = jax.random.uniform(k_x, (b, l, f), dtype)
    w1 = jax.random.normal(k_w1, (f, h), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (h,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (h, o), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (o,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    # MaxPool1d(kernel=seq_length, stride=1) over the full sequence is a max over axis 1.
    pooled = jnp.max(x, axis=1).reshape(-1, w1.shape[0])
    h = jax.nn.leaky_relu(pooled @ w1 + b1, negative_slope=CONFIG['negative_slope'])
    return h @ w2 + b2
