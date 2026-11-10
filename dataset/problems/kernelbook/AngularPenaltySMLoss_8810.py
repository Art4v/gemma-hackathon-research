"""[bf16 variant] AngularPenaltySMLoss (ArcFace): L2-normalize(dim=1) -> bias-free linear -> label-gathered arccos margin -> softmax-style loss -> mean."""
import jax
import jax.numpy as jnp

# batch == spatial == out_features is required: the torch module adds an
# (H, out, N) numerator to an (N, H, out) excluded-sum element by element.
CONFIG = {
    'name': 'AngularPenaltySMLoss_8810',
    'batch': 128,
    'channels': 32,
    'spatial': 128,
    'in_features': 128,
    'out_features': 128,
    's': 64.0,
    'm': 0.5,
    'eps': 1e-7,
    'norm_eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_l, k_w = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['spatial'], CONFIG['in_features']),
        dtype,
    )
    labels = jax.random.randint(k_l, (CONFIG['batch'],), 0, CONFIG['channels'], dtype=jnp.int32)
    w = jax.random.normal(k_w, (CONFIG['in_features'], CONFIG['out_features']), dtype) * 0.02
    return x, labels, w


def workload(x, labels, w):
    s, m, eps = CONFIG['s'], CONFIG['m'], CONFIG['eps']
    n, c = x.shape[0], x.shape[1]
    # F.normalize(x, p=2, dim=1); the module's weight normalization is a no-op.
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    xn = x / jnp.maximum(norm, CONFIG['norm_eps'])
    wf = xn @ w  # (N, C, H, out)

    # torch.diagonal(wf.transpose(0,1)[labels]) -> (H, out, N), [h,o,i] = wf[i, labels[i], h, o]
    target = wf[jnp.arange(n), labels]  # (N, H, out)
    cos_t = jnp.transpose(target, (1, 2, 0))
    numerator = s * jnp.cos(jnp.arccos(jnp.clip(cos_t, -1.0 + eps, 1.0 - eps)) + m)

    # Sum over C excluding each row's label column -> (N, H, out).
    mask = (jnp.arange(c)[None, :] == labels[:, None])[:, :, None, None]
    excl_sum = jnp.sum(jnp.where(mask, 0.0, jnp.exp(s * wf)), axis=1)

    denominator = jnp.exp(numerator) + excl_sum
    loss = numerator - jnp.log(denominator)
    return -jnp.mean(loss)
