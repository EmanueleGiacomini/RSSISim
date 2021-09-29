"""plot_utils.py
"""

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def build_generic_canvas(rows: int, cols: int, **kwargs):
    fig, axs = plt.subplots(rows, cols, **kwargs)
    return fig, axs


def draw_model_prediction(fig: plt.figure, ax: plt.axis, model: nn.Module, gw: int = 0,
                          min_point: (float, float) = (0, 0),
                          max_point: (float, float) = (100, 100), resolution: float = 1,
                          apply_ploss: bool = True, colorbar: bool = True ,*args, **kwargs):
    x_mesh, y_mesh = np.mgrid[min_point[0]:max_point[0]:resolution, min_point[1]:max_point[1]:resolution]
    x = x_mesh.ravel()
    y = y_mesh.ravel()
    data_x = np.stack([x, y], axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss).detach().numpy()[:, gw]
    z = z.reshape((int(max_point[0] - min_point[0]), int(max_point[1] - min_point[1]))).transpose()
    z[z < -200] = -200
    im = ax.imshow(z, norm=colors.PowerNorm(gamma=5))
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    return im


def draw_samples(ax: plt.axis, points, p_val=None, *args, **kwargs):
    scat = None
    p_val[p_val < -200] = -200
    if p_val is None:
        scat = ax.scatter(points[:, 0], points[:, 1], c='r', marker='x', alpha=0.4, *args, **kwargs)
    else:
        scat = ax.scatter(points[:, 0], points[:, 1], c=p_val, norm=colors.PowerNorm(gamma=5), marker='x', alpha=0.4,
                          *args, **kwargs)
    return scat


def draw_gw(ax: plt.axis, gw_pos, *args, **kwargs):
    return ax.scatter(gw_pos[0], gw_pos[1], c='r', marker='D')
