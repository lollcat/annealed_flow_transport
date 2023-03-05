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

"""An experiment setup for a two dimensional distribution."""

import ml_collections
ConfigDict = ml_collections.ConfigDict

from utils_fab import plot_contours_2D
import matplotlib.pyplot as plt

def plot_mog(samples, log_density):
  fig, ax = plt.subplots()
  plot_contours_2D(log_density, ax=ax, bound=60, levels=40)
  ax.plot(samples[:, 0], samples[:, 1], "o")



def get_config():
  """Returns a standard normal experiment config as ConfigDict."""
  config = ConfigDict()

  num_dim = 2

  config.seed = 1
  config.batch_size = 128
  config.estimation_batch_size = 2000
  config.sample_shape = (num_dim,)
  config.report_step = 1000
  config.vi_report_step = 50
  config.num_temps = 6  # Number of distributions, number of CRAFT inner iter is num_temps-1
  config.step_logging = False
  config.resample_threshold = 0.3
  config.stopping_criterion = 'time'
  config.use_resampling = True
  config.use_markov = True
  config.algo = 'craft'
  config.vi_iters = 1000
  config.vi_estimator = 'importance'
  config.save_checkpoint = True
  config.params_filename = "checkpoint"
  config.checkpoint_interval = 1000
  config.use_path_gradient = False
  config.use_plotting = True
  config.n_samples_plotting = config.batch_size
  config.plot = plot_mog

  optimization_config = ConfigDict()
  optimization_config.free_energy_iters = 1000
  optimization_config.aft_step_size = 1e-3
  optimization_config.vi_step_size = 1e-3
  config.craft_batch_size = 128
  config.craft_num_iters = int(10_000_000/config.craft_batch_size)
  optimization_config.craft_step_size = 1e-4

  config.optimization_config = optimization_config

  initial_config = ConfigDict()
  initial_config.density = 'MultivariateNormalDistribution'
  initial_config.shared_mean = 0.
  initial_config.diagonal_cov = 1.
  config.initial_config = initial_config

  final_config = ConfigDict()
  final_config.density = 'FABMoG'
  config.final_config = final_config

  flow_config = ConfigDict()
  flow_config.type = 'AffineInverseAutoregressiveFlow'
  flow_config.intermediate_hids_per_dim = 40
  flow_config.num_layers = 3  # total layers is this*(num_temps-1)
  flow_config.identity_init = True
  flow_config.bias_last = True

  config.flow_config = flow_config

  mcmc_config = ConfigDict()
  mcmc_config.rwm_steps_per_iter = 1
  rw_step_config = ConfigDict()
  rw_step_config.step_sizes = [5.0, 5.0]  # constant step size
  rw_step_config.step_times = [0., 1.]  # constant step size
  mcmc_config.rwm_step_config = rw_step_config


  mcmc_config.hmc_steps_per_iter = 0
  mcmc_config.slice_steps_per_iter = 0
  mcmc_config.nuts_steps_per_iter = 0
  config.mcmc_config = mcmc_config

  initial_sampler_config = ConfigDict()
  initial_sampler_config.initial_sampler = 'MultivariateNormalDistribution'
  initial_sampler_config.diagonal_cov =1.0
  config.initial_sampler_config = initial_sampler_config

  return config
