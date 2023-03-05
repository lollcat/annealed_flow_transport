from typing import Callable, Optional, Tuple

import itertools

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_contours(log_prob_func: Callable,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 20,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = jnp.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = jnp.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points)
    log_p_x = jnp.clip(log_p_x, a_min=log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points))
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def plot_marginal_pair(samples: chex.Array,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  alpha: float = 0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, a_min=bounds[0], a_max=bounds[1])
    samples = samples
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)


def get_target_log_prob_marginal_pair(log_prob, i: int, j: int, total_dim: int):
    def log_prob_marginal_pair(x_2d):
        x = jnp.zeros((x_2d.shape[0], total_dim))
        x = x.at[:, i].set(x_2d[:, 0])
        x = x.at[:, j].set(x_2d[:, 1])
        return log_prob(x)
    return log_prob_marginal_pair


def plot(samples, target_log_prob, n_rows=4, plotting_bounds=(-3, 3)):
    dim = samples.shape[-1]
    fig, axs = plt.subplots(n_rows, n_rows, sharex=True, sharey=True, figsize=(n_rows * 3, n_rows * 3))

    for i in range(n_rows):
        for j in range(n_rows):
            if i != j:
                log_prob_target = get_target_log_prob_marginal_pair(
                    target_log_prob, i, j, dim)
                plot_contours(log_prob_target, bounds=plotting_bounds, ax=axs[i, j], grid_width_n_points=40)
                plot_marginal_pair(samples, ax=axs[i, j], marginal_dims=(i, j), bounds=plotting_bounds, alpha=0.2)

            if j == 0:
                axs[i, j].set_xlabel(f"dim {i}")
            if i == dim - 1:
                axs[i, j].set_xlabel(f"dim {j}")
    plt.show()
