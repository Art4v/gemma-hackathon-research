"""[bf16 variant] ISAB: inducing-point multi-head attention blocks (linear projections, per-head scaled dot-product softmax attention, residual relu linear) chained as mab0(I, X) then mab1(X, H)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ISAB_16957',
    'batch': 8,
    'set_size': 4096,
    'dim_in': 512,
    'dim_out': 512,
    'num_heads': 4,
    'num_inds': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 18)
    dim_in = CONFIG['dim_in']
    dim_out = CONFIG['dim_out']
    X = jax.random.uniform(keys[0], (CONFIG['batch'], CONFIG['set_size'], dim_in), dtype)
    I = jax.random.normal(keys[1], (1, CONFIG['num_inds'], dim_out), dtype) * 0.02
    linear_shapes = [
        # mab0: fc_q, fc_k, fc_v, fc_o
        (dim_out, dim_out), (dim_in, dim_out), (dim_in, dim_out), (dim_out, dim_out),
        # mab1: fc_q, fc_k, fc_v, fc_o
        (dim_in, dim_out), (dim_out, dim_out), (dim_out, dim_out), (dim_out, dim_out),
    ]
    params = []
    for i, shape in enumerate(linear_shapes):
        w = jax.random.normal(keys[2 + 2 * i], shape, dtype) * 0.02
        b = jax.random.normal(keys[3 + 2 * i], (shape[1],), dtype) * 0.02
        params.extend([w, b])
    return (X, I, *params)


def _split_heads(x, num_heads):
    b, n, dim = x.shape
    return x.reshape(b, n, num_heads, dim // num_heads).transpose(0, 2, 1, 3)


def _mab(Q, K, params):
    q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b = params
    num_heads = CONFIG['num_heads']
    Q = Q @ q_w + q_b
    V = K @ v_w + v_b
    K = K @ k_w + k_b
    dim_V = Q.shape[-1]
    Q_ = _split_heads(Q, num_heads)
    K_ = _split_heads(K, num_heads)
    V_ = _split_heads(V, num_heads)
    A = jax.nn.softmax((Q_ @ jnp.swapaxes(K_, -1, -2)) / (dim_V ** 0.5), axis=-1)
    O = (Q_ + A @ V_).transpose(0, 2, 1, 3).reshape(Q.shape)
    return O + jax.nn.relu(O @ o_w + o_b)


def workload(*inputs):
    X, I = inputs[0], inputs[1]
    mab0_params = inputs[2:10]
    mab1_params = inputs[10:18]
    I_rep = jnp.tile(I, (X.shape[0], 1, 1))
    H = _mab(I_rep, X, mab0_params)
    return _mab(X, H, mab1_params)
