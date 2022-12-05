# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training for all SMC and flow algorithms."""
from jax.config import config
config.update("jax_enable_x64", True)
from typing import Callable, Tuple
# from annealed_flow_transport import craft
from annealed_flow_transport import densities
from annealed_flow_transport import flows
from annealed_flow_transport import samplers
from annealed_flow_transport import serialize
import annealed_flow_transport.aft_types as tp
import chex
import haiku as hk
import jax

import jax.numpy as jnp
import optax

from annealed_flow_transport import fab_craft as craft
from annealed_flow_transport.flow_with_inverse import make_realnvp_bijector
import distrax

# Type defs.
Array = jnp.ndarray
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
LogDensityByStep = tp.LogDensityByStep
RandomKey = tp.RandomKey
AcceptanceTuple = tp.AcceptanceTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
FreeEnergyEval = tp.FreeEnergyEval
MarkovKernelApply = tp.MarkovKernelApply
SamplesTuple = tp.SamplesTuple
LogWeightsTuple = tp.LogWeightsTuple
VfesTuple = tp.VfesTuple
InitialSampler = tp.InitialSampler
LogDensityNoStep = tp.LogDensityNoStep
assert_equal_shape = chex.assert_equal_shape
AlgoResultsTuple = tp.AlgoResultsTuple


def get_optimizer(initial_learning_rate: float,
                  boundaries_and_scales):
  """Get an optimizer possibly with learning rate schedule."""
  return optax.adam(initial_learning_rate)



def value_or_none(value: str,
                  config):
  if value in config:
    return config[value]
  else:
    return None



def prepare_outer_loop(
        flow_inverse_and_log_det: Callable[[Array], Tuple[Array, Array]],
        initial_sampler: InitialSampler,
                       initial_log_density: Callable[[Array], Array],
                       final_log_density: Callable[[Array], Array],
                       flow_func: Callable[[Array], Tuple[Array, Array]],
                       config) -> AlgoResultsTuple:
    """Shared code outer loops then calls the outer loops themselves.

    Args:
    initial_sampler: Function for producing initial sample.
    initial_log_density: Function for evaluating initial log density.
    final_log_density: Function for evaluating final log density.
    flow_func: Flow function to pass to Haiku transform.
    config: experiment configuration.
    Returns:
    An AlgoResultsTuple containing the experiment results.

    """
    key = jax.random.PRNGKey(config.seed)

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_inverse_fn = hk.without_apply_rng(hk.transform(flow_inverse_and_log_det))
    key, subkey = jax.random.split(key)
    single_normal_sample = initial_sampler(subkey,
                                         config.batch_size,
                                         config.sample_shape)
    key, subkey = jax.random.split(key)
    flow_init_params = flow_forward_fn.init(subkey, single_normal_sample)

    if value_or_none('save_checkpoint', config):
        def save_checkpoint(params):
            return serialize.save_checkpoint(config.params_filename, params)
    else:
        save_checkpoint = None

    opt = get_optimizer(
        config.optimization_config.craft_step_size,
        value_or_none('craft_boundaries_and_scales',
                      config.optimization_config))
    opt_init_state = opt.init(flow_init_params)
    log_step_output = None
    results = craft.outer_loop_craft(
        flow_inverse_fn,
        opt.update, opt_init_state,
        flow_init_params, flow_forward_fn.apply,
        initial_log_density, final_log_density,
        initial_sampler, key, config,
        log_step_output,
        save_checkpoint)
    return results


def run_experiment(config) -> AlgoResultsTuple:
  """Run a SMC flow experiment.

  Args:
    config: experiment configuration.
  Returns:
    An AlgoResultsTuple containing the experiment results.
  """
  log_density_initial = getattr(densities, config.initial_config.density)(
      config.initial_config, config.sample_shape[0])
  log_density_final = getattr(densities, config.final_config.density)(
      config.final_config, config.sample_shape[0])
  initial_sampler = getattr(samplers,
                            config.initial_sampler_config.initial_sampler)(
                                config.initial_sampler_config)

  make_flow = lambda: make_realnvp_bijector(x_ndim=config.sample_shape[0],
                                   flow_num_layers=10,
                                   mlp_hidden_size_per_x_dim=10,
                                   mlp_num_layers=2)

  def flow_func(x):
      flow: distrax.Bijector = make_flow()
      return jax.vmap(flow.forward_and_log_det)(x)

  def flow_func_inverse(y):
      flow: distrax.Bijector = make_flow()
      return jax.vmap(flow.inverse_and_log_det)(y)


  results = prepare_outer_loop(flow_func_inverse, initial_sampler, log_density_initial,
                               log_density_final, flow_func, config)
  import matplotlib.pyplot as plt
  from utils_fab import plot_contours_2D
  fig, ax = plt.subplots()
  plot_contours_2D(log_density_final, ax=ax, bound=60, levels=40)
  ax.plot(results.test_samples[:, 0], results.test_samples[:, 1], "o")
  plt.show()
  return results


if __name__ == '__main__':
    from configs.fab_mog import get_config
    exp_config = get_config()

    run_experiment(exp_config)
