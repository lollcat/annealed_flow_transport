# see https://github.com/deepmind/distrax/blob/master/examples/flow.py

from typing import Sequence, Tuple

import jax.nn

import distrax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

PRNGKey = chex.PRNGKey


def make_realnvp_bijector(
        x_ndim: int,
        flow_num_layers: int = 5,
        mlp_hidden_size_per_x_dim: int = 40,
        mlp_num_layers: int = 2,
):
        event_shape = (x_ndim,)  # is more general in jax example but here assume x is vector
        n_hidden_units = np.prod(event_shape) * mlp_hidden_size_per_x_dim
        flow = make_flow_model(
                event_shape=event_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[n_hidden_units] * mlp_num_layers,
        )
        return flow


def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int = 2) -> hk.Sequential:
    mlp =  hk.nets.MLP
    return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      mlp(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
    ])

def make_scalar_affine_bijector():
    def bijector_fn(params: jnp.ndarray):
      shift, log_scale = jnp.split(params, indices_or_sections=2, axis=-1)
      shift = jnp.squeeze(shift, axis=-1)
      log_scale = jnp.squeeze(log_scale, axis=-1)
      return distrax.ScalarAffine(shift=shift, log_scale=log_scale)
    return bijector_fn


def make_flow_model(event_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int]) -> distrax.Bijector:
    """Creates the flow model."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    event_ndims = len(event_shape)
    assert event_ndims == 1  # currently only focusing on this case (all elements in 1 dim).
    dim = event_shape[0]
    layers = []
    n_params = np.prod(event_shape)
    split_index = n_params // 2
    bijector_fn = make_scalar_affine_bijector()
    for i in range(num_layers):
        flip = i % 2 == 0
        if flip:
            num_bijector_params = n_params // 2
            num_dependent_params = n_params - num_bijector_params
        else:
            num_dependent_params = n_params // 2
            num_bijector_params = n_params - num_dependent_params

        layer = distrax.SplitCoupling(
            split_index=split_index,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape=(num_bijector_params,),
                                         hidden_sizes=hidden_sizes),
            event_ndims=event_ndims,
            swap=flip)
        layers.append(layer)

    flow = distrax.Chain(layers)
    return flow