"""[bf16 variant] KD_6200: soft-target cross_entropy + T-scaled log_softmax/softmax KL divergence (mean) -> alpha-weighted knowledge-distillation loss."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KD_6200',
    'batch': 8,
    'classes': 128,
    'height': 128,
    'width': 128,
    'class_axis': 1,
    'alpha': 4,
    'T': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['classes'], CONFIG['height'], CONFIG['width'])
    k_stu, k_tch, k_label = jax.random.split(jax.random.key(0), 3)
    output_stu = jax.random.uniform(k_stu, shape, dtype)
    output_tch = jax.random.uniform(k_tch, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    return output_stu, output_tch, label


def workload(output_stu, output_tch, label):
    axis = CONFIG['class_axis']
    alpha = CONFIG['alpha']
    T = CONFIG['T']

    # F.cross_entropy with class-probability targets, reduction='mean'.
    log_probs = jax.nn.log_softmax(output_stu, axis=axis)
    loss_stu = jnp.mean(-jnp.sum(label * log_probs, axis=axis))

    label_stu = jax.nn.log_softmax(output_stu / T, axis=axis)
    label_tch = jax.nn.softmax(output_tch / T, axis=axis)

    # F.kl_div, reduction='mean': xlogy(p, p) - p * log_q averaged over all elements.
    safe_tch = jnp.where(label_tch > 0, label_tch, jnp.ones_like(label_tch))
    pointwise = label_tch * jnp.log(safe_tch) - label_tch * label_stu
    loss_tch = jnp.mean(pointwise) * T * T

    return loss_stu * (1 - alpha) + loss_tch * alpha
