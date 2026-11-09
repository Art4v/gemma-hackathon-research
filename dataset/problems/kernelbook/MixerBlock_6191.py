"""MixerBlock: layernorm -> transpose -> MLP(fc1, gelu, fc2) -> transpose -> residual -> layernorm -> MLP -> residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MixerBlock_6191',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'tokens_dim': 512,
    'tokens_mlp_dim': 512,
    'channels_dim': 512,
    'channels_mlp_dim': 2048,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['tokens_dim']), dtype)
    tw1 = jax.random.normal(keys[1], (c['tokens_dim'], c['tokens_mlp_dim']), dtype) * 0.02
    tb1 = jax.random.normal(keys[2], (c['tokens_mlp_dim'],), dtype) * 0.02
    tw2 = jax.random.normal(keys[3], (c['tokens_mlp_dim'], c['tokens_dim']), dtype) * 0.02
    tb2 = jax.random.normal(keys[4], (c['tokens_dim'],), dtype) * 0.02
    cw1 = jax.random.normal(keys[5], (c['channels_dim'], c['channels_mlp_dim']), dtype) * 0.02
    cb1 = jax.random.normal(keys[6], (c['channels_mlp_dim'],), dtype) * 0.02
    cw2 = jax.random.normal(keys[7], (c['channels_mlp_dim'], c['channels_dim']), dtype) * 0.02
    cb2 = jax.random.normal(keys[8], (c['channels_dim'],), dtype) * 0.02
    return (x, tw1, tb1, tw2, tb2, cw1, cb1, cw2, cb2)


def _layer_norm_all(x):
    # F.layer_norm(x, x.shape[1:]) without affine: normalize over every non-batch axis.
    axes = tuple(range(1, x.ndim))
    mean = jnp.mean(x, axis=axes, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=axes, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps'])


def _mlp(x, w1, b1, w2, b2):
    h = jax.nn.gelu(x @ w1 + b1, approximate=False)
    return h @ w2 + b2


def workload(x, tw1, tb1, tw2, tb2, cw1, cb1, cw2, cb2):
    y = _layer_norm_all(x)
    y = jnp.swapaxes(y, 1, 2)
    y = _mlp(y, tw1, tb1, tw2, tb2)
    y = jnp.swapaxes(y, 1, 2)
    x = x + y
    y = _layer_norm_all(x)
    return x + _mlp(y, cw1, cb1, cw2, cb2)
