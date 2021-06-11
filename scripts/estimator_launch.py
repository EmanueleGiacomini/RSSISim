"""estimator_launch.py
"""

from scripts.model import RadioRSSIRegressor
import torch
from torch import nn as nn
import numpy as np
import pandas as pd
import copy

def read_data(fname: str):
    data_df = pd.read_csv(fname)
    data_pos = data_df[['x', 'y']].to_numpy().astype(np.float32)
    data_z = data_df[['recv_pwr0', 'recv_pwr1', 'recv_pwr2']].to_numpy().astype(np.float32)
    return data_pos, data_z

DATA_FILENAME = '../data/data.csv'
NO_EPOCHS = 40000

if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)

    x_data, y_data = read_data(DATA_FILENAME)
    # Take only first gateway
    y_data = y_data[:, 0]

    # Convert to torch tensors
    x_torch = torch.from_numpy(x_data)
    y_torch = torch.from_numpy(y_data)

    model = RadioRSSIRegressor(source_pos=torch.Tensor((0, 0)))
    optimizer = torch.optim.RMSprop(model.parameters(), 1e-3)
    loss_fn = nn.MSELoss()

    best_loss = np.Inf
    best_model = None

    for it in range(NO_EPOCHS):
        y_pred = model(x_torch)
        loss = loss_fn(y_pred, y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)
        if it % 10 == 0:
            print('it=', it)
            print(f'\tloss={loss}')
            print(f'\terror={y_pred - torch.resize_as_(y_torch, y_pred)}')

    torch.save(best_model, f'model.pth')
    exit(0)