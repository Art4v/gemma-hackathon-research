"""[bf16 variant] EnsembleModel: input normalize -> 7x batched ensemble matmul+bias (SiLU between) -> split mean/logvar -> softplus logvar clamp."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EnsembleModel_7391',
    'ensemble_size': 8,
    'batch': 8192,
    'input_num': 512,
    'hidden_num': 1024,
    'output_num': 256,
    'num_layers': 7,
    'weight_scale': 0.02,
    'max_logvar_init': 0.5,
    'min_logvar_init': -10.0,
}


def create_inputs(dtype=jnp.bfloat16):
    e = CONFIG['ensemble_size']
    n = CONFIG['batch']
    d_in = CONFIG['input_num']
    h = CONFIG['hidden_num']
    out = CONFIG['output_num']
    num_layers = CONFIG['num_layers']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 1 + 2 * num_layers)
    x = jax.random.uniform(keys[0], (n, d_in), dtype)
    inputs_mu = jnp.zeros((d_in,), dtype)
    inputs_sigma = jnp.ones((d_in,), dtype)

    dims = [d_in] + [h] * (num_layers - 1) + [2 * out]
    layers = []
    for i in range(num_layers):
        w = jax.random.normal(keys[1 + 2 * i], (e, dims[i], dims[i + 1]), dtype) * scale
        b = jax.random.normal(keys[2 + 2 * i], (e, 1, dims[i + 1]), dtype) * scale
        layers += [w, b]

    max_logvar = jnp.full((1, out), CONFIG['max_logvar_init'], dtype)
    min_logvar = jnp.full((1, out), CONFIG['min_logvar_init'], dtype)
    return (x, inputs_mu, inputs_sigma, *layers, max_logvar, min_logvar)


def workload(x, inputs_mu, inputs_sigma, *rest):
    layers = rest[:-2]
    max_logvar, min_logvar = rest[-2:]
    num_layers = len(layers) // 2

    h = (x - inputs_mu) / inputs_sigma
    for i in range(num_layers):
        # (N, in) @ (E, in, out) broadcasts to (E, N, out), matching torch.matmul.
        h = jnp.matmul(h, layers[2 * i]) + layers[2 * i + 1]
        if i < num_layers - 1:
            h = jax.nn.silu(h)

    out = max_logvar.shape[-1]
    mean = h[..., :out]
    logvar = h[..., out:]
    logvar = max_logvar - jax.nn.softplus(max_logvar - logvar)
    logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)
    return mean, logvar
