import jax.random

from annealed_flow_transport.flow_transport import *
from annealed_flow_transport.resampling import log_effective_sample_size

def update_samples_log_weights(
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    flow_params: FlowParams, samples: Samples, log_weights: Array,
    key: RandomKey, log_density: LogDensityByStep, step: int,
    use_markov: bool) -> Tuple[Array, Array, AcceptanceTuple]:
  """Update samples and log weights once the flow has been learnt."""
  use_resampling = True
  resample_threshold = 1.  # always resample
  transformed_samples, _ = flow_apply(flow_params, samples)
  assert_trees_all_equal_shapes(transformed_samples, samples)
  log_weights_new = reweight(log_weights, samples, flow_apply, flow_params,
                             log_density, step)
  assert_equal_shape([log_weights_new, log_weights])
  if use_resampling:
    subkey, key = jax.random.split(key)
    resampled_samples, log_weights_resampled = resampling.optionally_resample(
        subkey, log_weights_new, transformed_samples, resample_threshold)
    assert_trees_all_equal_shapes(resampled_samples, transformed_samples)
    assert_equal_shape([log_weights_resampled, log_weights_new])
  else:
    resampled_samples = transformed_samples
    log_weights_resampled = log_weights_new
  if use_markov:
    markov_samples, acceptance_tuple = markov_kernel_apply(
        step, key, resampled_samples)
  else:
    markov_samples = resampled_samples
    acceptance_tuple = (1., 1., 1.)
  return markov_samples, log_weights_new, acceptance_tuple


def get_ess_per_flow_step(initial_sampler, flow_apply, markov_kernel_apply,
                          flow_params_full, log_density,
                          key, config):
    key, subkey = jax.random.split(key)
    samples = initial_sampler(subkey, config.craft_batch_size,
                                      config.sample_shape)
    initial_log_weights = -jnp.log(config.craft_batch_size) * jnp.ones(
        config.craft_batch_size)
    ess_hist = []
    for i in range(config.num_temps - 1):
        flow_params = jax.tree_map(lambda x: x[i], flow_params_full)
        samples, next_log_weights, acceptance_tuple = update_samples_log_weights(
            flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
            flow_params=flow_params, samples=samples, log_weights=jnp.zeros_like(initial_log_weights),
            key=key, log_density=log_density, step=i + 1,
            use_markov=config.use_markov)
        ess = jnp.exp(log_effective_sample_size(next_log_weights)) / next_log_weights.shape[0]
        ess_hist.append(ess)
    return ess_hist



