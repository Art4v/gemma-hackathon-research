"""[bf16 variant] Mix_3516: swapaxes(1,2) -> LayerNorm -> channel MLP (Linear-GELU-Linear) -> swapaxes(1,2) residual, then LayerNorm -> MLP (Linear-GELU-Linear) residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Mix_3516',
    'batch': 8,
    'axis1': 128,
    'axis2': 128,
    'dim': 512,
    'y_dim': 512,
    'mlp_ratio': 4.0,
    'y_mlp_ratio': 4.0,
    'eps': 1e-5,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    dim = CONFIG['dim']
    y_dim = CONFIG['y_dim']
    y_hidden = int(y_dim / CONFIG['y_mlp_ratio'])
    hidden = int(dim * CONFIG['mlp_ratio'])
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 9)

    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], CONFIG['axis1'], CONFIG['axis2'], dim), dtype
    )

    norm1_gamma = jnp.ones((y_dim,), dtype)
    norm1_beta = jnp.zeros((y_dim,), dtype)
    channel_w1 = jax.random.normal(keys[1], (y_dim, y_hidden), dtype) * scale
    channel_b1 = jax.random.normal(keys[2], (y_hidden,), dtype) * scale
    channel_w2 = jax.random.normal(keys[3], (y_hidden, y_dim), dtype) * scale
    channel_b2 = jax.random.normal(keys[4], (y_dim,), dtype) * scale

    norm2_gamma = jnp.ones((dim,), dtype)
    norm2_beta = jnp.zeros((dim,), dtype)
    mlp_w1 = jax.random.normal(keys[5], (dim, hidden), dtype) * scale
    mlp_b1 = jax.random.normal(keys[6], (hidden,), dtype) * scale
    mlp_w2 = jax.random.normal(keys[7], (hidden, dim), dtype) * scale
    mlp_b2 = jax.random.normal(keys[8], (dim,), dtype) * scale

    return (
        x,
        norm1_gamma, norm1_beta,
        channel_w1, channel_b1, channel_w2, channel_b2,
        norm2_gamma, norm2_beta,
        mlp_w1, mlp_b1, mlp_w2, mlp_b2,
    )


def _layer_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + CONFIG['eps']) * gamma + beta


def _mlp(x, w1, b1, w2, b2):
    h = x @ w1 + b1
    h = jax.nn.gelu(h, approximate=False)
    return h @ w2 + b2


def workload(*inputs):
    (
        x,
        norm1_gamma, norm1_beta,
        channel_w1, channel_b1, channel_w2, channel_b2,
        norm2_gamma, norm2_beta,
        mlp_w1, mlp_b1, mlp_w2, mlp_b2,
    ) = inputs

    y = jnp.swapaxes(x, 1, 2)
    y = _mlp(_layer_norm(y, norm1_gamma, norm1_beta),
             channel_w1, channel_b1, channel_w2, channel_b2)
    x = x + jnp.swapaxes(y, 1, 2)

    x = x + _mlp(_layer_norm(x, norm2_gamma, norm2_beta),
                 mlp_w1, mlp_b1, mlp_w2, mlp_b2)
    return x
