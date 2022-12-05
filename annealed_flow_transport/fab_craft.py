# Copyright 2021 DeepMind Technologies Limited.
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

"""Continual Repeated Annealed Flow Transport (CRAFT) Monte Carlo algorithm."""
import time
from typing import Tuple, Callable

from absl import logging
from annealed_flow_transport import flow_transport
from annealed_flow_transport import markov_kernel
import annealed_flow_transport.aft_types as tp
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from annealed_flow_transport.flow_transport import reweight, assert_equal_shape, resampling


Array = jnp.ndarray
OptState = tp.OptState
UpdateFn = tp.UpdateFn
FlowParams = tp.FlowParams
FlowApply = tp.FlowApply
FlowInverseAndLogDetApply = Callable[[FlowParams, Array], Tuple[Array, Array]]
LogDensityNoStep = tp.LogDensityNoStep
InitialSampler = tp.InitialSampler
RandomKey = tp.RandomKey
SamplesTuple = tp.SamplesTuple
FreeEnergyAndGrad = tp.FreeEnergyAndGrad
MarkovKernelApply = tp.MarkovKernelApply
LogDensityByStep = tp.LogDensityByStep
AcceptanceTuple = tp.AcceptanceTuple
LogWeightsTuple = tp.LogWeightsTuple
AlgoResultsTuple = tp.AlgoResultsTuple
ParticleState = tp.ParticleState

# edited for fab
def update_samples_log_weights(
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    flow_params: FlowParams, samples: Array, log_weights: Array, key: RandomKey,
    log_density: LogDensityByStep, step: int, use_resampling: bool,
    use_markov: bool,
    resample_threshold: float) -> Tuple[Array, Array, AcceptanceTuple]:
  """Update samples and log weights once the flow has been learnt."""
  transformed_samples, _ = flow_apply(flow_params, samples)
  assert_equal_shape([transformed_samples, samples])
  log_weights_new = reweight(log_weights, samples, flow_apply, flow_params,
                             log_density, step)
  assert_equal_shape([log_weights_new, log_weights])
  if use_resampling:
    subkey, key = jax.random.split(key)
    resampled_samples, log_weights_resampled = resampling.optionally_resample(
        subkey, log_weights_new, transformed_samples, resample_threshold)
    assert_equal_shape([resampled_samples, transformed_samples])
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
  return markov_samples, log_weights_resampled, acceptance_tuple


def fab_loss_fn(flow_params, samples, log_weights, flow_inverse_and_log_det,
                log_density_by_step, i, log_w_ais):
    valid_indices = jnp.isfinite(log_weights) & jnp.isfinite(log_w_ais)
    w_adjust = jax.nn.softmax(log_weights + log_w_ais)
    # w_adjust = jnp.ones_like(w_adjust)
    # Remove samples with nan log prob under flow.

    w_adjust = jnp.where(valid_indices, w_adjust, 0)
    samples = jnp.where(valid_indices[:, None], samples, 0)

    z, log_det = flow_inverse_and_log_det.apply(flow_params, samples)
    base_log_prob = log_density_by_step(i-1, z)
    log_prob = base_log_prob + log_det

    return jnp.mean(w_adjust * log_prob)

def inner_step_craft(
    flow_inverse_and_log_det: FlowInverseAndLogDetApply,
    key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
    opt_update: UpdateFn, opt_state: OptState, flow_params: FlowParams,
    flow_apply: FlowApply, markov_kernel_apply: MarkovKernelApply,
    samples: Array, log_weights: Array, log_density: LogDensityByStep,
    step: int, config, fab_loss: bool = True
) -> Tuple[FlowParams, OptState, Array, Array, Array, Array, AcceptanceTuple]:
  """A temperature step of CRAFT.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    opt_update: function that updates the state of flow based on gradients etc.
    opt_state: state variables of the optimizer.
    flow_params: parameters of the flow.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    samples: Array containing samples.
    log_weights: Array containing train/validation/test log_weights.
    log_density:  function returning the log_density of a sample at given step.
    step: int giving current step of algorithm.
    config: experiment configuration.

  Returns:
    new_flow_params: New parameters of the flow.
    new_opt_state: New state variables of the optimizer.
    vfe: Value of the objective for this temperature.
    log_normalizer_increment: Scalar log of normalizing constant increment.
    next_samples: samples after temperature step has been performed.
    new_log_weights: log_weights after temperature step has been performed.
    acceptance_tuple: Acceptance rate of the Markov kernels used.
  """
  samples_in = samples
  vfe, flow_grads = free_energy_and_grad(flow_params,
                                         samples,
                                         log_weights,
                                         step)
  if not fab_loss:
      updates, new_opt_state = opt_update(flow_grads,
                                          opt_state)
  log_normalizer_increment = flow_transport.get_log_normalizer_increment(
      samples, log_weights, flow_apply, flow_params, log_density, step)
  next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
      flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
      flow_params=flow_params, samples=samples, log_weights=log_weights,
      key=key, log_density=log_density, step=step,
      use_resampling=False, use_markov=config.use_markov,
      resample_threshold=config.resample_threshold)

  if fab_loss:
      log_prob_x_0 = log_density(step-1, samples_in) - flow_apply(flow_params, samples_in)[1]
      z_x1, log_det_x1 = flow_inverse_and_log_det.apply(flow_params, samples)
      log_prob_x_1 = log_density(step-1, z_x1) + log_det_x1
      log_w_ais = log_density(step, samples_in) - log_prob_x_0 + \
                  log_density(step, samples) - log_prob_x_1
      loss, grad = jax.value_and_grad(fab_loss_fn)(
          flow_params, samples, log_weights, flow_inverse_and_log_det,
          log_density, step, log_w_ais
      )
      jax.debug.print("n valid log_w_ais {count}", count=jnp.sum(jnp.isfinite(log_w_ais)))
      jax.debug.print("n valid log weights {count}", count=jnp.sum(jnp.isfinite(log_weights)))
      jax.debug.print("grad-norm {grad_norm}", grad_norm=optax.global_norm(grad))
      # jax.debug.print("loss {loss}, log_w_ais:{log_w_ais}", loss=loss, log_w_ais=log_w_ais)
      updates, new_opt_state = opt_update(grad, opt_state)
  new_flow_params = optax.apply_updates(flow_params, updates)
  return new_flow_params, new_opt_state, vfe, log_normalizer_increment, next_samples, next_log_weights, acceptance_tuple


def inner_loop_craft(flow_inverse_and_log_det: FlowInverseAndLogDetApply,
                     key: RandomKey, free_energy_and_grad: FreeEnergyAndGrad,
                     opt_update: UpdateFn, opt_states: OptState,
                     transition_params: FlowParams, flow_apply: FlowApply,
                     markov_kernel_apply: MarkovKernelApply,
                     initial_sampler: InitialSampler,
                     log_density: LogDensityByStep, config):
  """Inner loop of CRAFT training.

  Uses Scan step requiring trees that have the same structure as the base input
  but with each leaf extended with an extra array index of size num_transitions.
  We call this an extended tree.

  Args:
    key: A JAX random key.
    free_energy_and_grad: function giving estimate of free energy and gradient.
    opt_update: function that updates the state of flow based on gradients etc.
    opt_states: Extended tree of optimizer states.
    transition_params: Extended tree of flow parameters.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    initial_sampler: A function that produces the initial samples.
    log_density: A function evaluating the log density for each step.
    config: A ConfigDict containing the configuration.
  Returns:
    final_samples: Array of final samples.
    final_log_weights: Array of final log_weights.
    final_transition_params: Extended tree of updated flow params.
    final_opt_states: Extended tree of updated optimizer parameters.
    overall_free_energy: Total variational free energy.
    log_normalizer_estimate: Estimate of the log normalizers.
  """
  subkey, key = jax.random.split(key)
  initial_samples = initial_sampler(subkey, config.craft_batch_size,
                                    config.sample_shape)
  initial_log_weights = -jnp.log(config.craft_batch_size) * jnp.ones(
      config.craft_batch_size)

  def scan_step(passed_state, per_step_input):
    samples, log_weights = passed_state
    flow_params, opt_state, key, inner_step = per_step_input
    new_flow_params, new_opt_state, vfe, log_normalizer_increment, next_samples, next_log_weights, acceptance_tuple = inner_step_craft(
        flow_inverse_and_log_det=flow_inverse_and_log_det, key=key, free_energy_and_grad=free_energy_and_grad,
        opt_update=opt_update, opt_state=opt_state, flow_params=flow_params,
        flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
        samples=samples, log_weights=log_weights, log_density=log_density,
        step=inner_step, config=config)
    next_passed_state = (next_samples, next_log_weights)
    per_step_output = (new_flow_params, new_opt_state, vfe,
                       log_normalizer_increment, acceptance_tuple)
    return next_passed_state, per_step_output

  initial_state = (initial_samples, initial_log_weights)
  inner_steps = jnp.arange(1, config.num_temps)
  keys = jax.random.split(key, config.num_temps-1)
  per_step_inputs = (transition_params, opt_states, keys, inner_steps)
  final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state,
                                               per_step_inputs)
  final_samples, final_log_weights = final_state
  final_transition_params, final_opt_states, free_energies, log_normalizer_increments, unused_acceptance_tuples = per_step_outputs
  overall_free_energy = jnp.sum(free_energies)
  log_normalizer_estimate = jnp.sum(log_normalizer_increments)
  return final_samples, final_log_weights, final_transition_params, final_opt_states, overall_free_energy, log_normalizer_estimate


def craft_evaluation_loop(key: RandomKey, transition_params: FlowParams,
                          flow_apply: FlowApply,
                          markov_kernel_apply: MarkovKernelApply,
                          initial_sampler: InitialSampler,
                          log_density: LogDensityByStep,
                          config) -> ParticleState:
  """A single pass of CRAFT with fixed flows.

  Uses Scan step requiring trees that have the same structure as the base input
  but with each leaf extended with an extra array index of size num_transitions.
  We call this an extended tree.

  Args:
    key: A JAX random key.
    transition_params: Extended tree of flow parameters.
    flow_apply: function that applies the flow.
    markov_kernel_apply: function that applies the Markov transition kernel.
    initial_sampler: A function that produces the initial samples.
    log_density: A function evaluating the log density for each step.
    config: A ConfigDict containing the configuration.
  Returns:
    final_samples: Array of final samples.
    final_log_weights: Array of final log_weights.
    log_normalizer_estimate: Estimate of the log normalizers.
  """
  subkey, key = jax.random.split(key)
  initial_samples = initial_sampler(subkey, config.craft_batch_size,
                                    config.sample_shape)
  initial_log_weights = -jnp.log(config.craft_batch_size) * jnp.ones(
      config.craft_batch_size)

  def scan_step(passed_state, per_step_input):
    samples, log_weights = passed_state
    flow_params, key, inner_step = per_step_input
    log_normalizer_increment = flow_transport.get_log_normalizer_increment(
        samples, log_weights, flow_apply, flow_params, log_density, inner_step)
    next_samples, next_log_weights, acceptance_tuple = flow_transport.update_samples_log_weights(
        flow_apply=flow_apply, markov_kernel_apply=markov_kernel_apply,
        flow_params=flow_params, samples=samples, log_weights=log_weights,
        key=key, log_density=log_density, step=inner_step,
        use_resampling=config.use_resampling, use_markov=config.use_markov,
        resample_threshold=config.resample_threshold)

    next_passed_state = (next_samples, next_log_weights)
    per_step_output = (log_normalizer_increment, acceptance_tuple)
    return next_passed_state, per_step_output

  initial_state = (initial_samples, initial_log_weights)
  inner_steps = jnp.arange(1, config.num_temps)
  keys = jax.random.split(key, config.num_temps-1)
  per_step_inputs = (transition_params, keys, inner_steps)
  final_state, per_step_outputs = jax.lax.scan(scan_step, initial_state,
                                               per_step_inputs)
  final_samples, final_log_weights = final_state
  log_normalizer_increments, unused_acceptance_tuples = per_step_outputs
  log_normalizer_estimate = jnp.sum(log_normalizer_increments)
  particle_state = ParticleState(
      samples=final_samples,
      log_weights=final_log_weights,
      log_normalizer_estimate=log_normalizer_estimate)
  return particle_state


def outer_loop_craft(
                    flow_inverse_and_log_det: FlowInverseAndLogDetApply,
                    opt_update: UpdateFn,
                     opt_init_state: OptState,
                     flow_init_params: FlowParams,
                     flow_apply: FlowApply,
                     initial_log_density: LogDensityNoStep,
                     final_log_density: LogDensityNoStep,
                     initial_sampler: InitialSampler,
                     key: RandomKey,
                     config,
                     log_step_output,
                     save_checkpoint) -> AlgoResultsTuple:
  """Outer loop for CRAFT training.

  Args:
    opt_update: function that updates the state of flow based on gradients etc.
    opt_init_state: initial state variables of the optimizer.
    flow_init_params: initial state of the flow.
    flow_apply: function that applies the flow.
    initial_log_density: The log density of the starting distribution.
    final_log_density: The log density of the target distribution.
    initial_sampler: A function that produces the initial samples.
    key: A Jax random key.
    config: A ConfigDict containing the configuration.
    log_step_output: Callable that logs the step output.
    save_checkpoint: None or function that takes params and saves them.
  Returns:
    An AlgoResults tuple containing a summary of the results.
  """
  num_temps = config.num_temps
  density_by_step = flow_transport.GeometricAnnealingSchedule(
      initial_log_density, final_log_density, num_temps)
  markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
      config.mcmc_config, density_by_step, num_temps)
  def free_energy_short(flow_params: FlowParams,
                        samples: Array,
                        log_weights: Array,
                        step: int) -> Array:
    return flow_transport.transport_free_energy_estimator(samples,
                                                          log_weights,
                                                          flow_apply,
                                                          flow_params,
                                                          density_by_step,
                                                          step)

  free_energy_and_grad = jax.value_and_grad(free_energy_short)
  def short_inner_loop(rng_key: RandomKey,
                       curr_opt_states: OptState,
                       curr_transition_params):
    return inner_loop_craft(flow_inverse_and_log_det=flow_inverse_and_log_det,
                            key=rng_key,
                            free_energy_and_grad=free_energy_and_grad,
                            opt_update=opt_update,
                            opt_states=curr_opt_states,
                            transition_params=curr_transition_params,
                            flow_apply=flow_apply,
                            markov_kernel_apply=markov_kernel_by_step,
                            initial_sampler=initial_sampler,
                            log_density=density_by_step,
                            config=config)
  inner_loop_jit = jax.jit(short_inner_loop)

  repeater = lambda x: jnp.repeat(x[None], num_temps-1, axis=0)
  opt_states = jax.tree_util.tree_map(repeater, opt_init_state)
  transition_params = jax.tree_util.tree_map(repeater, flow_init_params)

  logging.info('Performing initial step redundantly for accurate timing...')
  initial_start_time = time.time()
  inner_loop_jit(key, opt_states, transition_params)
  initial_finish_time = time.time()
  initial_time_diff = initial_finish_time - initial_start_time
  logging.info('Initial step time / seconds  %f: ', initial_time_diff)
  logging.info('Launching training...')

  start_time = time.time()
  for step in tqdm(range(config.craft_num_iters)):
    key, subkey = jax.random.split(key)
    final_samples, final_log_weights, transition_params, opt_states, overall_free_energy, log_normalizer_estimate = inner_loop_jit(
        subkey, opt_states, transition_params)
    if step % config.report_step == 0:
      if log_step_output is not None:
        delta_time = time.time()-start_time
        log_step_output(step=step,
                        training_objective=overall_free_energy,
                        log_normalizer_estimate=log_normalizer_estimate,
                        delta_time=delta_time,
                        samples=final_samples,
                        log_weights=final_log_weights)
      info = f'Step {step}: Free energy {overall_free_energy} Log Normalizer estimate {log_normalizer_estimate}'
      logging.info(
          'Step %05d: Free energy %f Log Normalizer estimate %f',
          step, overall_free_energy, log_normalizer_estimate
          )
      print(info)
      import matplotlib.pyplot as plt
      from utils_fab import plot_contours_2D
      fig, ax = plt.subplots()
      plot_contours_2D(final_log_density, ax=ax, bound=60, levels=40)
      ax.plot(final_samples[:, 0], final_samples[:, 1], "o")
      plt.show()

  finish_time = time.time()
  delta_time = finish_time - start_time
  logging.info('Delta time / seconds  %f: ', delta_time)
  logging.info('Log normalizer estimate %f: ', log_normalizer_estimate)
  if save_checkpoint:
    save_checkpoint(transition_params)
  results = AlgoResultsTuple(
      test_samples=final_samples,
      test_log_weights=final_log_weights,
      log_normalizer_estimate=log_normalizer_estimate,
      delta_time=delta_time,
      initial_time_diff=initial_time_diff)

  import matplotlib.pyplot as plt
  from utils_fab import plot_contours_2D
  fig, ax = plt.subplots()
  plot_contours_2D(final_log_density, ax=ax, bound=60, levels=40)
  ax.plot(final_samples[:, 0], final_samples[:, 1], "o")
  plt.show()
  return results
