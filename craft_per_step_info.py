import jax.random

from annealed_flow_transport.flow_transport import *
from annealed_flow_transport.resampling import log_effective_sample_size

def update_samples_log_weights_with_info(
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    flow_params: FlowParams, samples: Samples, log_weights: Array,
    key: RandomKey, log_density: LogDensityByStep, step: int,
    use_resampling: bool, use_markov: bool,
    resample_threshold: float) -> Tuple[Array, Array, AcceptanceTuple, int, float]:
  """Update samples and log weights once the flow has been learnt."""
  transformed_samples, _ = flow_apply(flow_params, samples)
  assert_trees_all_equal_shapes(transformed_samples, samples)
  log_weights_new = reweight(log_weights, samples, flow_apply, flow_params,
                             log_density, step)
  assert_equal_shape([log_weights_new, log_weights])
  if use_resampling:
    subkey, key = jax.random.split(key)
    indices = jax.random.categorical(key, log_weights,
                                     shape=(log_weights.shape[0],))
    n_unique_samples = jnp.unique(indices).shape[0]
    ess = jnp.exp(log_effective_sample_size(log_weights_new)) / log_weights_new.shape[0]
    resampled_samples, log_weights_resampled = resampling.optionally_resample(
        subkey, log_weights_new, transformed_samples, resample_threshold)
    assert_trees_all_equal_shapes(resampled_samples, transformed_samples)
    assert_equal_shape([log_weights_resampled, log_weights_new])
  else:
    raise Exception # Undesired use.
  if use_markov:
    markov_samples, acceptance_tuple = markov_kernel_apply(
        step, key, resampled_samples)
  else:
    raise Exception # Undersired use.
  return markov_samples, log_weights_resampled, acceptance_tuple, n_unique_samples, ess


def get_craft_info_per_step(initial_sampler, flow_apply, markov_kernel_apply,
                          flow_params_full, log_density,
                          key, config):
  key, subkey = jax.random.split(key)
  samples = initial_sampler(subkey, config.craft_batch_size,
                            config.sample_shape)
  log_weights = -jnp.log(config.craft_batch_size) * jnp.ones(
    config.craft_batch_size)
  n_unique_samples_hist = []
  ess_hist = []
  for i in range(config.num_temps - 1):
    flow_params = jax.tree_map(lambda x: x[i], flow_params_full)
    samples, log_weights, acceptance_tuple, n_unique_samples, ess = update_samples_log_weights_with_info(
      flow_apply=jax.jit(flow_apply), markov_kernel_apply=jax.jit(markov_kernel_apply),
      flow_params=flow_params, samples=samples, log_weights=log_weights,
      key=key, log_density=log_density, step=i + 1,
      use_markov=config.use_markov, use_resampling=config.use_resampling,
      resample_threshold=config.resample_threshold)
    n_unique_samples_hist.append(n_unique_samples)
    ess_hist.append(ess)
  info = {}
  info.update(n_unique_samples_hist=n_unique_samples_hist, log_weights=log_weights, ess=ess_hist)
  return info


