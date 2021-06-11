"""model.py
"""

import torch
from torch import nn as nn
import numpy as np


class RadioRSSIRegressor(nn.Module):
    def __init__(self, n_layers: int = 4, h_dim: int = 10, source_pos: torch.Tensor = torch.Tensor([0, 0]),
                 wl: float = 0.1206):
        """

        :param n_layers: number of hidden layers to be used
        :param h_dim: dimension of each hidden layer (Latent space dimension)
        :param source_pos: Cartesian tensor representing the position of the gateway
        :param wl: Wavelength of the radio wave expressed in meters [m]
        """
        super(RadioRSSIRegressor, self).__init__()
        self.n_layers = n_layers
        self.layer_lst = nn.ModuleList(
            [nn.Linear(2, h_dim)] +
            [nn.Linear(h_dim, h_dim) for _ in range(n_layers)] +
            [nn.Linear(h_dim, 1)])
        self.source_pos = source_pos
        self.wl = wl
        ...

    def xy2rt(self, x: torch.Tensor) -> torch.Tensor:
        """ Converts the input feature Tensor x containing cartesian coordinates of sampled points into
        a feature Tensor containing a polar representation of sampled points related to source_pos.
        input = [x, y]
        x_source, y_source = x-source_pos.x, y-source_pos.y
        output.x = sqrt(x_source*x_source + y_source*y_source)
        output.y = atan2(y_source, x_source)
        """
        x_source, y_source = x[:, 0] - self.source_pos[0], x[:, 1] - self.source_pos[1]
        r = torch.sqrt(x_source * x_source + y_source * y_source)
        theta = torch.atan2(y_source, x_source)
        return torch.stack((r, theta))

    def freespace_loss(self, r) -> torch.Tensor:
        """ Computes the free-space loss given the x_polar coordinates. To compute the loss, the function only
        requires the range from the source_pos
        """
        return -20 * torch.log10(4 * np.pi * r / self.wl)

    def forward(self, x):
        # Convert input to polar coordinates
        x_polar = self.xy2rt(x)
        x_polar = torch.transpose(x_polar, 1, 0)
        out = nn.ReLU()(self.layer_lst[0](x_polar))
        for i in range(1, self.n_layers - 1):
            out = nn.ReLU()(self.layer_lst[i](out))
        out = self.layer_lst[-1](out)
        # out represents the obstacle residual estimated loss
        # Augment out with the estimated free-space loss
        fs_loss = torch.resize_as_(self.freespace_loss(x_polar[:, 0]), out)
        return fs_loss + out


if __name__ == '__main__':
    print('RadioRSSIRegressor test')
    gateway_position = torch.Tensor((10, 15))
    print('Gateway position=', gateway_position)
    model = RadioRSSIRegressor(source_pos=gateway_position)

    sample_x = np.float32([0.0, 10, 20, 30, 40, 50, 10])
    sample_y = np.float32([0.0, 20, 10, 50, 30, 40, 16])

    sample_xy = torch.from_numpy(np.stack((sample_x, sample_y), axis=1))
    sample_polar = model.xy2rt(sample_xy)
    print('Sample in cartesian coordinates=', sample_xy)
    print('Sample in polar coordinates=', sample_polar)
    print('Model output test=', model(sample_xy))
    exit(0)
