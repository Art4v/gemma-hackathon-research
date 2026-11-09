"""D_phiVpsi: two MLPs (Linear+LeakyReLU x3) on x and y, then sum((phi_x @ W) * psi_y, axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'D_phiVpsi_7775',
    'batch': 16384,
    'x_dim': 2048,
    'y_dim': 1,
    'phi_layers': [1024, 1024, 512],
    'psi_layers': [1024, 1024, 512],
    'negative_slope': 0.2,
}


def _mlp_params(key, in_dim, sizes, dtype):
    params = []
    dims = [in_dim] + sizes
    keys = jax.random.split(key, 2 * len(sizes))
    for i in range(len(sizes)):
        w = jax.random.normal(keys[2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 * i + 1], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return params


def create_inputs(dtype=jnp.float32):
    kx, ky, kphi, kpsi, kw = jax.random.split(jax.random.key(0), 5)
    b = CONFIG['batch']
    x = jax.random.uniform(kx, (b, CONFIG['x_dim']), dtype)
    y = jax.random.uniform(ky, (b, CONFIG['y_dim']), dtype)
    phi = _mlp_params(kphi, CONFIG['x_dim'], CONFIG['phi_layers'], dtype)
    psi = _mlp_params(kpsi, CONFIG['y_dim'], CONFIG['psi_layers'], dtype)
    # torch inits W with randn (unit scale), kept here.
    W = jax.random.normal(kw, (CONFIG['phi_layers'][-1], CONFIG['psi_layers'][-1]), dtype)
    return (x, y, *phi, *psi, W)


def _mlp(h, params):
    slope = CONFIG['negative_slope']
    for i in range(0, len(params), 2):
        h = h @ params[i] + params[i + 1]
        h = jnp.where(h >= 0, h, slope * h)
    return h


def workload(x, y, pw0, pb0, pw1, pb1, pw2, pb2, qw0, qb0, qw1, qb1, qw2, qb2, W):
    b = x.shape[0]
    x = x.reshape(b, -1)
    y = y.reshape(b, 1)
    phi_x = _mlp(x, (pw0, pb0, pw1, pb1, pw2, pb2))
    psi_y = _mlp(y, (qw0, qb0, qw1, qb1, qw2, qb2))
    return jnp.sum((phi_x @ W) * psi_y, axis=1, keepdims=True)
