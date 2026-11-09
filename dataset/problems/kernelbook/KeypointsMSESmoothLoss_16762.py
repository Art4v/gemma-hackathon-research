"""[bf16 variant] KeypointsMSESmoothLoss: weighted squared heatmap error, pow-0.1 smoothing above threshold, normalized sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KeypointsMSESmoothLoss_16762',
    'batch': 16,
    'num_joints': 128,
    'height': 128,
    'width': 128,
    'threshold': 400.0,
    'output_scale': 20.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_out, k_tgt, k_w = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch'], CONFIG['num_joints'], CONFIG['height'], CONFIG['width'])
    # Spread out the predictions so that some weighted squared errors exceed the
    # threshold and the smoothing branch actually runs.
    output = jax.random.normal(k_out, shape, dtype) * CONFIG['output_scale']
    target = jax.random.uniform(k_tgt, shape, dtype)
    target_weight = jax.random.uniform(k_w, (CONFIG['batch'], CONFIG['num_joints']), dtype)
    return output, target, target_weight


def workload(output, target, target_weight):
    thr = CONFIG['threshold']
    b, j = output.shape[0], output.shape[1]
    pred = output.reshape(b, j, -1)
    gt = target.reshape(b, j, -1)
    dimension = pred.shape[-1]
    # Same broadcasting as torch's target_weight[..., None].
    diff = (gt - pred) ** 2 * target_weight[..., None]
    diff = jnp.where(diff > thr, jnp.power(diff, 0.1) * thr ** 0.9, diff)
    denom = dimension * jnp.maximum(1.0, jnp.sum(target_weight))
    return jnp.sum(diff) / denom
