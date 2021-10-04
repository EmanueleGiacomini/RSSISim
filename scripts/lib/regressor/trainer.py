"""trainer.py
"""


from ..plot_utils import draw_model_prediction, draw_samples

from copy import deepcopy
import time
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from matplotlib import pyplot as plt


def _progress_cb(cval, fval, title, total_time, loss, val_loss, best_loss):
    n_bars = 30
    ratio = cval / fval
    ratio *= n_bars
    ratio = int(ratio)
    print('\r' + '=' * ratio + '>' + ' ' * (n_bars - ratio) + '|' +
          f'{total_time:.3f}s {title}, epoch={cval}, loss={loss:.3f}, val_loss={val_loss:.3f}, best_loss={best_loss:.3f}', end='')
    if cval == fval:
        print('\nDone.')
    return

class TorchDataHandler():
    @staticmethod 
    def parse_data(data_x: np.float32, data_y: np.float32, batch_size, train_test_split) -> (DataLoader, DataLoader):
        torch_x = torch.from_numpy(data_x)
        torch_y = torch.from_numpy(data_y)

        dataset = TensorDataset(torch_x, torch_y)
        train_size = int(train_test_split * len(dataset))
        test_size = int(len(dataset) - train_size)
        print('train_size: ', train_size, ' test_size: ', test_size)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        return train_loader, test_loader


class Trainer():
    @staticmethod
    def train(train_x: np.float32, train_y: np.float32, model: nn.Module, optimizer: str, lr: float, epochs: int,
              best_fit: bool = True, early_stop: int = -1, plot_training: bool = False, train_test_split = 0.6, batch_size=64,
              cuda_en: bool = True, detection_flag: bool = False, *args, **kwargs) -> (nn.Module, np.float32, np.float32):


        use_cuda = torch.cuda.is_available() and cuda_en
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        torch.backends.cudnn.benchmark = True

        model.to(device)


        optim = None
        if optimizer == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
        elif optimizer == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
        else:
            raise Exception(f'{optimizer} is not a valid optimizer')

        if detection_flag is False:
            loss_fn = nn.MSELoss(reduction='mean').to(device)
        else:
            # Main idea:
            # loss = k1 * MSE(Rssi_pred, Rssi_gt) * Detect_gt + k2 * BCE(Detect_pred, Detect_gt)
            rssi_loss_fn = nn.MSELoss(reduction='none').to(device)
            detect_loss_fn = nn.BCELoss(reduction='none').to(device)

        if plot_training:
            fig, axs = plt.subplots(1, train_y.shape[1], figsize=(10, 10))

        # Best fit and early stop variables
        best_loss = np.inf
        best_model_state_dict = None
        early_stop_ctr = 0

        train_loader, test_loader = TorchDataHandler.parse_data(train_x, train_y, batch_size, train_test_split)

        train_loss = np.zeros((epochs,))
        train_val_loss = np.zeros((epochs,))
        
        # Training loop
        t0 = time.perf_counter()
        for it in range(epochs):
            # Training
            batch_t_loss = 0
            for local_batch, local_labels in train_loader:
                optim.zero_grad()
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                pred_batch = model(local_batch)
                if detection_flag is False:
                    loss = loss_fn(pred_batch, local_labels)
                    loss.backward()
                else:
                    loss = 10 * rssi_loss_fn(pred_batch[:,:,0], local_labels[:,:,0]) * local_labels[:,:,1] + \
                        detect_loss_fn(torch.round(pred_batch[:,:,1]), torch.round(local_labels[:,:,1]))
                    loss = loss.sum() / batch_size
                    loss.backward()
                optim.step()
                batch_t_loss += loss.cpu().float()
                
            batch_t_loss /= len(train_loader)
            
            # Validation
            batch_v_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, local_labels in test_loader:
                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                    pred_batch = model(local_batch)
                    if detection_flag is False:
                        loss = loss_fn(pred_batch, local_labels)
                    else:
                        
                        loss = rssi_loss_fn(pred_batch[:,:,0], local_labels[:,:,0]) * local_labels[:,:,1] + \
                            detect_loss_fn(torch.round(pred_batch[:,:,1]), torch.round(local_labels[:,:,1]))
                        loss = loss.mean()    
                    batch_v_loss += loss.cpu().float()
            
            batch_v_loss /= len(test_loader)

            train_loss[it] = batch_t_loss
            train_val_loss[it] = batch_v_loss

            if plot_training:
                if it % 1 == 0:
                    for gw in range(train_y.shape[1]):
                        axs[gw].clear()
                        draw_model_prediction(fig, axs[gw], model, gw, colorbar=False, resolution=1, 
                            detection_flag=detection_flag, device=device)
                        draw_samples(axs[gw], train_x, p_val=train_y[:, gw, :], detection_flag=detection_flag)
                    plt.pause(.01)
                    plt.draw()
            
            # Get best model
            if best_fit:
                if batch_v_loss < best_loss:
                    best_loss = batch_v_loss
                    best_model_state_dict = {k:v.cpu() for k, v in model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

            t1 = time.perf_counter() - t0
            _progress_cb(it + 1, epochs, 'Training', t1, batch_t_loss, batch_v_loss, best_loss)
        if best_fit:
            model.cpu()
            model.load_state_dict(best_model_state_dict)
        
        return model, train_loss, train_val_loss