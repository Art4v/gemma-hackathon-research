"""Linear (matmul + bias) -> GroupNorm -> LeakyReLU -> elementwise x + x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '62_Matmul_GroupNorm_LeakyReLU_Sum',
    'batch_size': 1024,
    'input_size': 8192,
    'hidden_size': 8192,
    'num_groups': 512,
    'eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (CONFIG['batch_size'], CONFIG['input_size']), dtype)
    # nn.Linear weight passed as (in, out) so that workload uses x @ w.
    w = jax.random.normal(k_w, (CONFIG['input_size'], CONFIG['hidden_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['hidden_size'],), dtype) * 0.02
    gamma = jnp.ones((CONFIG['hidden_size'],), dtype)
    beta = jnp.zeros((CONFIG['hidden_size'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    num_groups = CONFIG['num_groups']
    eps = CONFIG['eps']
    slope = CONFIG['negative_slope']

    h = x @ w + b  # (N, C)
    n, c = h.shape
    hg = h.reshape(n, num_groups, c // num_groups)
    mean = jnp.mean(hg, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(hg - mean), axis=-1, keepdims=True)
    hn = ((hg - mean) * jax.lax.rsqrt(var + eps)).reshape(n, c)
    y = hn * gamma + beta
    y = jnp.where(y >= 0, y, slope * y)
    return y + y
