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

"""An experiment setup for the Cox process posterior distribution."""
import ml_collections
ConfigDict = ml_collections.ConfigDict


def get_config():
  """Returns a standard normal experiment config as ConfigDict."""
  config = ConfigDict()

  config.seed = 1
  config.batch_size = 2000
  config.estimation_batch_size = 2000
  config.sample_shape = (32*32,)
  config.report_step = 1
  config.vi_report_step = 100
  config.num_layers = 1
  config.step_logging = False
  config.num_temps = 11
  config.resample_threshold = 0.3
  config.write_samples = True
  config.stopping_criterion = 'time'
  config.use_resampling = True
  config.use_markov = True
  config.use_path_gradient = False
  config.algo = 'craft'
  config.craft_num_iters = 200
  config.snf_num_iters = 1000
  config.craft_batch_size = 2000
  config.snf_batch_size = 2000
  config.vi_iters = 100000
  config.checkpoint_interval = 200000
  config.vi_estimator = 'importance'

  optimization_config = ConfigDict()
  optimization_config.free_energy_iters = 500
  optimization_config.aft_step_size = 1e-2
  optimization_config.craft_step_size = 5e-2
  optimization_config.craft_boundaries_and_scales = ({100: 1e-2},)
  optimization_config.snf_step_size = 0.1
  optimization_config.snf_boundaries_and_scales = ({70: 5e-2,
                                                    100: 1e-2},)
  optimization_config.vi_step_size = 1e-4
  config.optimization_config = optimization_config

  initial_config = ConfigDict()
  initial_config.density = 'MultivariateNormalDistribution'
  initial_config.shared_mean = 0.
  initial_config.diagonal_cov = 1.
  config.initial_config = initial_config

  final_config = ConfigDict()
  final_config.density = 'LogGaussianCoxPines'
  final_config.use_whitened = False
  final_config.file_path = ''
  config.final_config = final_config

  flow_config = ConfigDict()
  flow_config.type = 'DiagonalAffine'
  flow_config.sample_shape = config.sample_shape
  config.flow_config = flow_config
  initial_sampler_config = ConfigDict()
  initial_sampler_config.initial_sampler = 'MultivariateNormalDistribution'
  config.initial_sampler_config = initial_sampler_config

  mcmc_config = ConfigDict()
  hmc_step_config = ConfigDict()
  hmc_step_config.step_times = [0., 0.25, 0.5, 1.]
  hmc_step_config.step_sizes = [0.3, 0.3, 0.2, 0.2]
  nuts_step_config = ConfigDict()
  nuts_step_config.step_times = [0., 0.25, 0.5, 1.]
  nuts_step_config.step_sizes = [0.7, 0.7, 0.5, 0.5]

  mcmc_config.hmc_step_config = hmc_step_config
  mcmc_config.slice_step_config = hmc_step_config
  mcmc_config.nuts_step_config = nuts_step_config
  mcmc_config.hmc_steps_per_iter = 1
  mcmc_config.use_jax_hmc = True
  mcmc_config.rwm_steps_per_iter = 0
  mcmc_config.hmc_num_leapfrog_steps = 10

  mcmc_config.slice_steps_per_iter = 0
  mcmc_config.nuts_steps_per_iter = 0
  mcmc_config.slice_max_doublings = 5
  mcmc_config.nuts_max_tree_depth = 4
  config.mcmc_config = mcmc_config
  config.save_params = False

  return config
