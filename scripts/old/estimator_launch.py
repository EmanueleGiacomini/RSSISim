"""estimator_launch.py
"""

from scripts.old.model import RadioRSSIRegressor
from scripts.old.omnet_preproc import parse_omnet
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

def read_data_pd(data_df: pd.DataFrame, replace_nan=np.nan):
    data_pos = data_df[['x', 'y']].to_numpy().astype(np.float32)
    gw_keys = []#[k if 'recv_pwr' in k for k in data_df.keys()]
    for k in data_df.keys():
        if 'recv_pwr' in k:
            gw_keys.append(k)
    data_z = data_df[gw_keys].replace(to_replace='none', value=replace_nan).to_numpy().astype(np.float32)
    return data_pos, data_z

DATA_FILENAME = '../../data/data.csv'
NO_EPOCHS = 40000

if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)

    #x_data, y_data = read_data(DATA_FILENAME)
    data_df, gw_pos = parse_omnet('../data/test-0617-wall-final.csv', [np.float32((240, 240)), np.float32((200, 140))])
    #data_df, gw_pos = parse_omnet('../data/omnet00.csv')
    x_data, y_data = read_data_pd(data_df, replace_nan=-1000)
    # Take only first gateway
    y_data = y_data[:, 0]
    y_data[y_data < -1000] = -1000

    # Convert to torch tensors
    x_torch = torch.from_numpy(x_data)
    y_torch = torch.from_numpy(y_data)

    model = RadioRSSIRegressor(source_pos=torch.from_numpy(gw_pos[0]), wl=0.34538302)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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

    torch.save(best_model, f'model_omnet_gate0.pth')
    exit(0)