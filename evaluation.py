from annealed_flow_transport.serialize import load_checkpoint
from annealed_flow_transport.densities import FABMoG
from annealed_flow_transport.craft import craft_evaluation_loop, ParticleState, \
    flow_transport
from annealed_flow_transport import markov_kernel
from annealed_flow_transport import densities
from annealed_flow_transport import flows
from annealed_flow_transport import samplers
import haiku as hk
import jax
import jax.numpy as jnp
from jax.config import config
# config.update("jax_enable_x64", True)

from ess_per_step import get_ess_per_flow_step
from craft_per_step_info import get_craft_info_per_step


def get_flow_init_params(config, key=jax.random.PRNGKey(0)):
    initial_sampler, log_density_initial, log_density_final, flow_func = \
        setup_basic_objects(config)
    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_init_params = flow_forward_fn.init(key, jnp.zeros((1, *config.sample_shape)))
    repeater = lambda x: jnp.repeat(x[None], config.num_temps - 1, axis=0)
    transition_params = jax.tree_util.tree_map(repeater, flow_init_params)
    return transition_params


def evaluate_mog(forward_pass_function, n_runs=5):
    target = FABMoG(config, (2, ))
    key = jax.random.PRNGKey(0)

    # collect
    x_s = []
    log_w_s = []
    for i in range(n_runs):
        key, subkey = jax.random.split(key)
        particle_state: ParticleState = forward_pass_function(subkey)
        x_s.append(particle_state.samples)
        log_w_s.append(particle_state.log_weights)
    x = jnp.concatenate(x_s, axis=0)
    log_w = jnp.concatenate(log_w_s, axis=0)

    # run evaluate
    eval_info = target.eval(x=x, log_w=log_w)
    return eval_info

def evaluate_many_well(forward_pass_function, n_runs=5):
    key = jax.random.PRNGKey(0)

    # collect
    x_s = []
    log_w_s = []
    for i in range(n_runs):
        key, subkey = jax.random.split(key)
        particle_state: ParticleState = forward_pass_function(subkey)
        x_s.append(particle_state.samples)
        log_w_s.append(particle_state.log_weights)
    log_w = jnp.stack(log_w_s, axis=0)
    x = jnp.stack(x_s, axis=0)

    # run evaluate
    ess_s = 1 / jnp.sum(jax.nn.softmax(log_w, axis=-1) ** 2, axis=-1) / log_w.shape[-1]
    info = {}
    info.update(ess=ess_s, x=x, log_w=log_w)
    return info


def setup_basic_objects(config):
    """Copied from train.py
    """
    log_density_initial = getattr(densities, config.initial_config.density)(
        config.initial_config, config.sample_shape)
    log_density_final = getattr(densities, config.final_config.density)(
        config.final_config, config.sample_shape)
    initial_sampler = getattr(samplers,
                              config.initial_sampler_config.initial_sampler)(
        config.initial_sampler_config)

    def flow_func(x):
        flow = getattr(flows, config.flow_config.type)(config.flow_config)
        return flow(x)

    return initial_sampler, log_density_initial, log_density_final, flow_func


def make_forward_pass_func(config, transition_params, eval_batch_size=int(1000)):
    config.craft_batch_size = eval_batch_size
    num_temps = config.num_temps

    initial_sampler, log_density_initial, log_density_final, flow_func = \
        setup_basic_objects(config)

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_apply = flow_forward_fn.apply
    log_density_by_step = flow_transport.GeometricAnnealingSchedule(
        log_density_initial, log_density_final, num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
        config.mcmc_config, log_density_by_step, num_temps)

    @jax.jit
    def forward_pass(key) -> ParticleState:
        particle_state = craft_evaluation_loop(
            key=key,
            transition_params=transition_params,
            flow_apply=flow_apply,
            markov_kernel_apply=markov_kernel_by_step,
            initial_sampler=initial_sampler,
            log_density = log_density_by_step,
            config = config
        )
        return particle_state

    return forward_pass

def make_get_ess(config, transition_params, eval_batch_size=int(1000)):
    config.craft_batch_size = eval_batch_size
    num_temps = config.num_temps

    initial_sampler, log_density_initial, log_density_final, flow_func = \
        setup_basic_objects(config)

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_apply = flow_forward_fn.apply
    log_density_by_step = flow_transport.GeometricAnnealingSchedule(
        log_density_initial, log_density_final, num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
        config.mcmc_config, log_density_by_step, num_temps)

    # @jax.jit
    def get_ess_per_layer_fn(key) -> ParticleState:
        ess_per_layer = get_ess_per_flow_step(initial_sampler, flow_apply, markov_kernel_by_step,
                          transition_params, log_density_by_step,
                          key, config)
        return ess_per_layer

    return get_ess_per_layer_fn


def make_get_resample_info(config, transition_params, eval_batch_size=int(1000)):
    config.craft_batch_size = eval_batch_size
    num_temps = config.num_temps

    initial_sampler, log_density_initial, log_density_final, flow_func = \
        setup_basic_objects(config)

    flow_forward_fn = hk.without_apply_rng(hk.transform(flow_func))
    flow_apply = flow_forward_fn.apply
    log_density_by_step = flow_transport.GeometricAnnealingSchedule(
        log_density_initial, log_density_final, num_temps)
    markov_kernel_by_step = markov_kernel.MarkovTransitionKernel(
        config.mcmc_config, log_density_by_step, num_temps)

    # @jax.jit
    def get_craft_info_per_step_fn(key) -> ParticleState:
        ess_per_layer = get_craft_info_per_step(initial_sampler, flow_apply, markov_kernel_by_step,
                          transition_params, log_density_by_step,
                          key, config)
        return ess_per_layer

    return get_craft_info_per_step_fn



if __name__ == '__main__':
    mw = True
    if mw:
        from configs.many_well import get_config
        filename = "checkpoint_craft_mw"
    else:
        from configs.fab_mog import get_config
        filename = "checkpoint_old"

    config = get_config()
    config.use_resampling = True
    config.use_markov = True
    transition_params = load_checkpoint(filename)
    forward_pass_function = make_forward_pass_func(config, transition_params=transition_params)
    get_ess_per_layer_fn = make_get_ess(config=config, transition_params=transition_params,
                                        eval_batch_size=100)
    get_resample_info_fn = make_get_resample_info(config=config, transition_params=transition_params,
                                        eval_batch_size=100)

    if mw:
        resample_info = get_resample_info_fn(jax.random.PRNGKey(0))
        ess_per_step = get_ess_per_layer_fn(jax.random.PRNGKey(0))
        eval_info = evaluate_many_well(forward_pass_function)
    else:
        eval_info = evaluate_mog(forward_pass_function)
    print(eval_info)
