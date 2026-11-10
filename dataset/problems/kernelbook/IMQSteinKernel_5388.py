"""[bf16 variant] IMQ Stein kernel: pairwise squared diffs -> per-dim median bandwidth -> exp(beta * log(alpha + sum(sq / bw)))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'IMQSteinKernel_5388',
    'num_particles': 256,
    'dim': 32768,
    'alpha': 0.5,
    'beta': -0.5,
    'd_chunk': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['num_particles'], CONFIG['dim'])
    X = jax.random.uniform(kx, shape, dtype)
    Y = jax.random.uniform(ky, shape, dtype)
    return X, Y


def workload(X, Y):
    n, d = X.shape
    m = Y.shape[0]
    if n != m:
        raise ValueError(f'IMQSteinKernel median bandwidth needs X and Y with equal rows, got {n} and {m}')
    alpha, beta = CONFIG['alpha'], CONFIG['beta']
    # ponytail: largest divisor of d not above d_chunk, so small test shapes also work
    chunk = next(c for c in range(min(CONFIG['d_chunk'], d), 0, -1) if d % c == 0)
    rows, cols = jnp.triu_indices(m, k=1)  # rows index Y (i), cols index X (j > i)
    num_pairs = rows.shape[0]
    log_np = jnp.log(jnp.asarray(m + 1, X.dtype))
    xt = X.T.reshape(d // chunk, chunk, n)
    yt = Y.T.reshape(d // chunk, chunk, m)

    def step(acc, xy):
        xc, yc = xy
        pair = (xc[:, cols] - yc[:, rows]) ** 2  # (chunk, K)
        # torch.median returns the lower median
        median = jnp.sort(pair, axis=-1)[:, (num_pairs - 1) // 2]
        bw = median / log_np
        sq = (xc[:, None, :] - yc[:, :, None]) ** 2  # (chunk, M, N): [i, j] = X[j] - Y[i]
        return acc + jnp.sum(sq / bw[:, None, None], axis=0), None

    total, _ = lax.scan(step, jnp.zeros((m, n), X.dtype), (xt, yt))
    base = alpha + total
    return jnp.exp(beta * jnp.log(base))
