"""model_plotter.py
"""

import torch
from torch import nn as nn
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scripts.old.estimator_launch import read_data_pd
from scripts.old.omnet_preproc import parse_omnet


def plot_model_map(model: nn.Module, val_x: torch.Tensor, val_y: np.array, x_shape):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    z = model(val_x, apply_ploss=True).detach().numpy()
    z = z.reshape(x_shape)
    z[z < -90] = -90
    val_y = val_y.reshape(x_shape)
    val_y[val_y < -90] = -90
    ax[0].set_title('Model Estimate')
    ax[0].imshow(z)
    ax[1].set_title('Ground Truth')
    ax[1].imshow(val_y)
    plt.show()


def plot_models(model_lst: [nn.Module], val_x: torch.Tensor, val_y: [np.array], x_shape):
    num_models = len(model_lst)
    assert num_models == len(val_y)
    fig, ax = plt.subplots(num_models, 2, figsize=(10, 10))
    for i in range(num_models):
        model = model_lst[i]
        z_i = model(val_x).detach().numpy()
        ax[i, 0].set_title(f'Model_{i} Estimate')
        ax[i, 0].imshow(z_i.reshape(x_shape))
        ax[i, 1].set_title('Ground Truth')
        y = val_y[i].reshape(x_shape)
        y[y < -90] = -90
        ax[i, 1].imshow(y)
    plt.tight_layout()
    plt.show()


def plot_omnet_model(model: nn.Module, min_p: np.float32, max_p: np.float32, step: float, ax=None,
                     train_x: np.array = None):
    X_mesh, Y_mesh = np.mgrid[min_p[0]:max_p[0]:step, min_p[1]:max_p[1]:step]
    X = X_mesh.ravel()
    Y = Y_mesh.ravel()
    data_x = np.stack([X, Y], axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss=True).detach().numpy()
    z = z.reshape((int(max_p[0] - min_p[0]), int(max_p[1] - min_p[1])))
    if ax is None:
        plt.imshow(z)
        if train_x is not None:
            plt.scatter(train_x[:, 0], train_x[:, 1])
            plt.plot(train_x[:, 0], train_x[:, 1], '--k')
        plt.colorbar()
        plt.show()
    else:
        im = ax.imshow(z, norm=colors.PowerNorm(gamma=5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if train_x is not None:
            ax.scatter(train_x[:, 0], train_x[:, 1], c='r', marker='x')
            ax.plot(train_x[:, 0], train_x[:, 1], '--k')
        fig.colorbar(im, cax=cax, orientation='vertical')


if __name__ == '__main__':
    # x_data, y_data = read_data('./data/data.csv')
    data_df, gw_pos = parse_omnet('./data/test-0617-wall-final.csv')
    x_data, y_data = read_data_pd(data_df, replace_nan=-1000)
    # Take only first gateway
    y_data = [y_data[:, 0], y_data[:, 1]]
    x_torch = torch.from_numpy(x_data)

    img_size = int(np.sqrt(x_data.shape[0]))

    model_lst = [torch.load('./scripts/model_gate0.pth'),
                 torch.load('./scripts/model_gate1.pth'),
                 torch.load('./scripts/model_gate2.pth')]

    # plot_models(model_lst, x_torch, y_data, (img_size, img_size))
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plot_omnet_model(torch.load('./scripts/model_omnet_gate0.pth'),
                     np.float32((0, 0)), np.float32((500, 500)), 1, axs[0])
    plot_omnet_model(torch.load('./scripts/model_omnet_gate1.pth'),
                     np.float32((0, 0)), np.float32((500, 500)), 1, axs[1])
    plt.show()
    exit(0)
