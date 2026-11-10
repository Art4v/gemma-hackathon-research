"""[bf16 variant] ISAB: tile inducing points -> MAB(I, X) -> MAB(X, H), each MAB = linear q/k/v -> multi-head softmax attention -> residual relu linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ISAB_4910',
    'batch': 8,
    'num_points': 4096,
    'dim_in': 512,
    'dim_out': 512,
    'num_heads': 4,
    'num_inds': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    num_points = CONFIG['num_points']
    dim_in = CONFIG['dim_in']
    dim_out = CONFIG['dim_out']
    num_inds = CONFIG['num_inds']
    scale = CONFIG['weight_scale']

    weight_shapes = [
        # mab0: fc_q, fc_k, fc_v, fc_o
        (dim_out, dim_out), (dim_in, dim_out), (dim_in, dim_out), (dim_out, dim_out),
        # mab1: fc_q, fc_k, fc_v, fc_o
        (dim_in, dim_out), (dim_out, dim_out), (dim_out, dim_out), (dim_out, dim_out),
    ]
    keys = jax.random.split(jax.random.key(0), 2 + 2 * len(weight_shapes))

    X = jax.random.uniform(keys[0], (batch, num_points, dim_in), dtype)
    I = jax.random.normal(keys[1], (1, num_inds, dim_out), dtype) * scale

    params = []
    for i, shape in enumerate(weight_shapes):
        w = jax.random.normal(keys[2 + 2 * i], shape, dtype) * scale
        b = jax.random.normal(keys[3 + 2 * i], (shape[1],), dtype) * scale
        params.extend([w, b])
    return (X, I, *params)


def _linear(x, w, b):
    return x @ w + b


def _split_heads(x, num_heads):
    b, n, d = x.shape
    return jnp.transpose(x.reshape(b, n, num_heads, d // num_heads), (0, 2, 1, 3))


def _merge_heads(x):
    b, h, n, d = x.shape
    return jnp.transpose(x, (0, 2, 1, 3)).reshape(b, n, h * d)


def _mab(Q, K, params):
    w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o = params
    num_heads = CONFIG['num_heads']

    q = _linear(Q, w_q, b_q)
    k = _linear(K, w_k, b_k)
    v = _linear(K, w_v, b_v)
    dim_V = q.shape[-1]

    q_h = _split_heads(q, num_heads)
    k_h = _split_heads(k, num_heads)
    v_h = _split_heads(v, num_heads)

    scores = (q_h @ jnp.swapaxes(k_h, -1, -2)) / (dim_V ** 0.5)
    A = jax.nn.softmax(scores, axis=-1)
    O = _merge_heads(q_h + A @ v_h)
    return O + jax.nn.relu(_linear(O, w_o, b_o))


def workload(*inputs):
    X, I = inputs[0], inputs[1]
    mab0_params = inputs[2:10]
    mab1_params = inputs[10:18]
    I_rep = jnp.tile(I, (X.shape[0], 1, 1))
    H = _mab(I_rep, X, mab0_params)
    return _mab(X, H, mab1_params)
