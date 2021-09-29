"""regressor.py
"""

import torch
from torch import nn
import numpy as np

from ..io_utils import DataReader, FileType


class SingleGWRegressor(nn.Module):
    def __init__(self, n_layers: int = 10, h_dim: int = 10, source_pos: torch.Tensor = torch.Tensor([0, 0]),
                 wl: float = 0.1206, tx_pwr: float = 2+3*np.random.uniform(0, 4), polar=False):
        """

        :param n_layers: number of hidden layers to be used
        :param h_dim: dimension of each hidden layer (Latent space dimension)
        :param source_pos: Cartesian tensor representing the position of the gateway
        :param wl: Wavelength of the radio wave expressed in meters [m]
        """
        super(SingleGWRegressor, self).__init__()
        self.n_layers = n_layers
        self.layer_lst = nn.ModuleList(
            [nn.Linear(2, h_dim)] +
            [nn.Linear(h_dim, h_dim) for _ in range(n_layers)] +
            [nn.Linear(h_dim, 1)])
        self.source_pos = source_pos
        self.wl = wl
        self.tx_pwr = tx_pwr
        self.polar = polar

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

    def global2local(self, x: torch.Tensor) -> torch.Tensor:
        """ Converts the input feature tensor x containing cartesian absolute values of sampled points into
        a feature Tensor containing cartesian local coordinates of sampled points related to source_pos
        input = [x, y]
        output.x = x-source_pos.x
        output.y = y-source_pos.y
        """
        x_local, y_local = x[:, 0] - self.source_pos[0], x[:, 1] - self.source_pos[1]
        return torch.stack((x_local, y_local))

    def freespace_loss(self, r) -> torch.Tensor:
        """ Computes the free-space loss given the x_polar coordinates. To compute the loss, the function only
        requires the range from the source_pos
        """
        return -20 * torch.log10(4 * np.pi * r / self.wl)

    def forward(self, x, apply_ploss=True):
        # Convert input to polar coordinates
        x_polar = self.xy2rt(x)
        x_polar = torch.transpose(x_polar, 1, 0)
        x_local = self.global2local(x)
        x_local = torch.transpose(x_local, 1, 0)
        if self.polar:
            out = nn.ReLU()(self.layer_lst[0](x_polar))
        else:
            out = nn.ReLU()(self.layer_lst[0](x_local))
        
        for i in range(1, self.n_layers - 1):
            out = nn.ReLU()(self.layer_lst[i](out))
        out = self.layer_lst[-1](out)
        # out represents the obstacle residual estimated loss
        # Augment out with the estimated free-space loss
        fs_loss = torch.resize_as_(self.freespace_loss(x_polar[:, 0]), out)
        if apply_ploss:
            return fs_loss + out
        return out


class MultiGWRegressor(nn.Module):
    def __init__(self, n_gw: int, gw_pos: [np.float32], *args, **kwargs):
        super(MultiGWRegressor, self).__init__()
        self.n_gw = n_gw
        self.submodule_lst = nn.ModuleList([SingleGWRegressor(**kwargs, source_pos=gw_pos[i]) for i in range(n_gw)])

    def forward(self, x, apply_ploss=True):
        z = torch.zeros(x.shape[0], self.n_gw).to(next(self.parameters()).device)
        for i in range(self.n_gw):
            z[:, i] = self.submodule_lst[i](x, apply_ploss)[:, -1]
        return z


def test_MultiGWRegressor():
    print('MultiGWRegressor test')
    gw_pos = DataReader.read_gw_info('../data/data_gw.txt')
    data_x, data_y, gw_pos = DataReader.read('../data/data.csv', FileType.RSSISIM, gw_pos)
    model = MultiGWRegressor(len(gw_pos), gw_pos, wl=1e-4)
    x_torch = torch.from_numpy(data_x)
    y_torch = torch.from_numpy(data_y)
    print(nn.MSELoss(reduction='sum')(model(x_torch), y_torch))

