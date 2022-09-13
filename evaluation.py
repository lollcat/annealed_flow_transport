from annealed_flow_transport.serialize import load_checkpoint
from annealed_flow_transport.densities import FABMoG
from annealed_flow_transport.craft import craft_evaluation_loop, ParticleState, \
    flow_transport, markov_kernel
from annealed_flow_transport import densities
from annealed_flow_transport import flows
from annealed_flow_transport import samplers
from configs.fab_mog import get_config
import haiku as hk
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


def evaluate(forward_pass_function, n_runs=5):
    target = FABMoG(config, 2)
    key = jax.random.PRNGKey(0)

    # collect
    x_s = []
    log_w_s = []
    for i in range(n_runs):
        particle_state: ParticleState = forward_pass_function(key)
        x_s.append(particle_state.samples)
        log_w_s.append(particle_state.log_weights)
    x = jnp.concatenate(x_s, axis=0)
    log_w = jnp.concatenate(log_w_s, axis=0)

    # run evaluate
    eval_info = target.eval(x=x, log_w=log_w)
    return eval_info


def setup_basic_objects(config):
    """Copied from train.py
    """
    log_density_initial = getattr(densities, config.initial_config.density)(
      config.initial_config, config.sample_shape[0])
    log_density_final = getattr(densities, config.final_config.density)(
      config.final_config, config.sample_shape[0])
    initial_sampler = getattr(samplers,
                            config.initial_sampler_config.initial_sampler)(
                                config.initial_sampler_config)

    def flow_func(x):
        flow = getattr(flows, config.flow_config.type)(config.flow_config)
        return jax.vmap(flow)(x)

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



if __name__ == '__main__':
    filename = "checkpoint_old"
    config = get_config()
    transition_params = load_checkpoint(filename)
    forward_pass_function = make_forward_pass_func(config, transition_params=transition_params)
    eval_info = evaluate(forward_pass_function)
    print(eval_info)