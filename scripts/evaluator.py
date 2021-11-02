"""evaluator.py
"""

from numpy.lib.function_base import kaiser
from lib.io_utils import DataReader, FileType, LoadFromFile
from lib.plot_utils import *
import torch

import os
import pandas as pd
from matplotlib import pyplot as plt



def cross01_compare_loss(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_title('Validation loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    it = 1
    for k in data_dict.keys():
        if 'cross01' not in k:
            continue
        # Moving mean to reduce noise
        ax.plot(data_dict[k]['rss_loss'], 'C'+str(it), alpha=0.3)
        ax.plot(data_dict[k]['rss_loss'].rolling(10).mean(), 'C'+str(it), label=k)
        it += 1
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_compare_mae(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Validation MAE')
    ax.set_xlabel('epochs')
    ax.set_ylabel('MAE')
    it = 1
    for k in data_dict.keys():
        if 'cross01' not in k:
            continue
        # Moving mean to reduce noise
        ax.plot(data_dict[k]['mae'], 'C'+str(it), alpha=0.3)
        ax.plot(data_dict[k]['mae'].rolling(10).mean(), 'C'+str(it), label=k)
        it += 1
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_compare_mse(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('Validation MSE')
    ax.set_xlabel('epochs')
    ax.set_ylabel('MSE')
    it = 1
    for k in data_dict.keys():
        if 'cross01' not in k:
            continue
        # Moving mean to reduce noise
        ax.plot(data_dict[k]['mse'], 'C'+str(it), alpha=0.3)
        ax.plot(data_dict[k]['mse'].rolling(10).mean(), 'C'+str(it), label=k)
        it += 1
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_compare_loss_no_mod(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['cross01_2_5']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['cross01_2_5']['rss_loss'].rolling(10).mean(), 'C0', label='cross01_2_5')
    ax.plot(data_dict['cross01_5_5']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['cross01_5_5']['rss_loss'].rolling(10).mean(), 'C1', label='cross01_5_5')
    ax.plot(data_dict['cross01_30_10']['rss_loss'], 'C2', alpha=0.3)
    ax.plot(data_dict['cross01_30_10']['rss_loss'].rolling(10).mean(), 'C2', label='cross01_30_10')
    ax.plot(data_dict['cross01_10_30']['rss_loss'], 'C3', alpha=0.3)
    ax.plot(data_dict['cross01_10_30']['rss_loss'].rolling(10).mean(), 'C3', label='cross01_10_30')
    ax.plot(data_dict['cross01_30_30']['rss_loss'], 'C4', alpha=0.3)
    ax.plot(data_dict['cross01_30_30']['rss_loss'].rolling(10).mean(), 'C4', label='cross01_30_30')

    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_compare_bn_dp(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['cross01_10_30']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['cross01_10_30']['rss_loss'].rolling(10).mean(), 'C0', label='cross01_10_30')
    ax.plot(data_dict['cross01_10_30_bn']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['cross01_10_30_bn']['rss_loss'].rolling(10).mean(), 'C1', label='cross01_10_30_bn')
    ax.plot(data_dict['cross01_10_30_dp']['rss_loss'], 'C2', alpha=0.3)
    ax.plot(data_dict['cross01_10_30_dp']['rss_loss'].rolling(10).mean(), 'C2', label='cross01_10_30_dp')
    ax.plot(data_dict['cross01_10_30_bn_dp']['rss_loss'], 'C3', alpha=0.3)
    ax.plot(data_dict['cross01_10_30_bn_dp']['rss_loss'].rolling(10).mean(), 'C3', label='cross01_10_30_bn_dp')
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_compare_polar(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['cross01_10_30']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['cross01_10_30']['rss_loss'].rolling(10).mean(), 'C0', label='cross01_10_30')
    ax.plot(data_dict['cross01_10_30_polar']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['cross01_10_30_polar']['rss_loss'].rolling(10).mean(), 'C1', label='cross01_10_30_polar')
    ax.set_yscale('log')
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def cross01_ground_truth(data_path: str, obstacle_path: str, gw_info: str, key: str='recv_pwr0') -> plt.Figure:
    data_df = pd.read_csv(data_path)
    if key not in data_df.keys():
        raise Exception(f'The key {key} is not present in {data_path} db.')
    z = np.reshape(data_df[key].to_numpy(), (100, 100)).transpose()
    z[z < -148] = -148

    gw_data = DataReader.read_gw_info(gw_info)
    
    
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(z, norm=colors.PowerNorm(gamma=2))
    for gw_idx in gw_data:
        draw_gw(ax, gw_idx)
    with open(obstacle_path) as f:
        for line in f.readlines():
            obst_pts = [float(x) for x in line.strip().split(',')]
            draw_obstacle(ax, obst_pts)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Ground truth')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical', label='dBm')
    return fig

def cross01_ground_truth_polar(data_path: str, obstalce_path: str, gw_info: str, gw_ref: int=0) -> plt.Figure:
    fig = plt.figure(figsize=(10, 5))
    
    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)
    data_df = pd.read_csv(data_path)
    key = 'recv_pwr'+str(gw_ref)
    if key not in data_df.keys():
        raise Exception(f'The key {key} is not present in {data_path} db.')

    gw_data = DataReader.read_gw_info(gw_info)
    gw_pos = gw_data[gw_ref]
    gw_x, gw_y = gw_pos[0], gw_pos[1]
    
    theta_lst = []
    rad_lst = []
    z_lst = []

    for _, row in data_df.iterrows():
        px, py, z = row['x'], row['y'], row[key]
        px = px - gw_x
        py = py - gw_y
        theta_lst.append(np.arctan2(py, px))
        rad_lst.append(np.sqrt(px**2 + py**2))
        z_lst.append(z)
    ax.scatter(rad_lst, theta_lst, c=z_lst)
    ax.set_xlabel('radial distance [m]')
    ax.set_ylabel('azimuth [rad]')
    ax.set_title(f'Polar view')

    ax1.imshow(data_df[key].to_numpy().reshape((100,100)).transpose(), aspect='auto')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title(f'Cartesian view')
    return fig

def cross01_prediction(model_path: str, data_path: str, obstacle_path: str, gw_info: str, gw_ref: int=0, confidence: float=0.6) -> plt.Figure:
    fig, ax0 = plt.subplots(1, 1, figsize=(5,5))
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.set_title('RSSI')

    model = torch.load(model_path)
    x_mesh, y_mesh = np.mgrid[0:100:1, 0:100:1]
    x, y = x_mesh.ravel(), y_mesh.ravel()
    data_x = np.stack((x, y), axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss=True).detach()[:, gw_ref, :].cpu().numpy()
    z_conf = z[:, 1]
    z_val = z[:, 0]
    z_val[z_conf < confidence] = -148

    z_val = z_val.reshape((100, 100)).transpose()
    z_conf = z_conf.reshape((100, 100)).transpose()

    ax0.imshow(z_val)

    gw_data = DataReader.read_gw_info(gw_info)
    for gw_idx in gw_data:
        draw_gw(ax0, gw_idx)

    with open(obstacle_path) as f:
        for line in f.readlines():
            obst_pts = [float(x) for x in line.strip().split(',')]
            draw_obstacle(ax0, obst_pts)

    ax0.set_xlim(0, 99)
    ax0.set_ylim(0, 99)

    return fig

def cross01_prediction_vs_gt(model_path: str, data_path: str, obstacle_path: str, gw_info: str, gw_ref: int=0, confidence: float=0.6) -> plt.Figure:
    
    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.set_title('Ground Truth')
    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Prediction')
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)

    key = f'recv_pwr{gw_ref}'
    z = pd.read_csv(data_path)[key].to_numpy().reshape((100, 100)).transpose()
    z[z<-148] = -148
    gt_im = ax0.imshow(z)

    ## Model
    model = torch.load(model_path)
    x_mesh, y_mesh = np.mgrid[0:100:1, 0:100:1]
    x, y = x_mesh.ravel(), y_mesh.ravel()
    data_x = np.stack((x, y), axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss=True).detach()[:, gw_ref, :].cpu().numpy()
    z_conf = z[:, 1]
    z_val  = z[:, 0]
    z_val[z_conf < confidence] = -148

    z_val = z_val.reshape((100, 100)).transpose()
    z_conf = z_conf.reshape((100, 100)).transpose()
    
    pred_im = ax1.imshow(z_val)
    #pred_im = ax1.imshow(z_conf)

    gw_data = DataReader.read_gw_info(gw_info)

    for gw_idx in gw_data:
        draw_gw(ax0, gw_idx)
        draw_gw(ax1, gw_idx)
    with open(obstacle_path) as f:
        for line in f.readlines():
            obst_pts = [float(x) for x in line.strip().split(',')]
            draw_obstacle(ax0, obst_pts)
            draw_obstacle(ax1, obst_pts)

    divider = make_axes_locatable(ax0)
    cax0 = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(gt_im, cax=cax0, orientation='vertical', label='dBm')

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(pred_im, cax=cax1, orientation='vertical', label='dBm')


    return fig


def omnet_compare_loss(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_title('Validation loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    it = 1
    for k in data_dict.keys():
        if 'cross01' in k:
            continue
        # Moving mean to reduce noise
        ax.plot(data_dict[k]['rss_loss'], 'C'+str(it), alpha=0.3)
        ax.plot(data_dict[k]['rss_loss'].rolling(10).mean(), 'C'+str(it), label=k)
        it += 1
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    return fig

def omnet_compare_loss_no_mod(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['omnet_2_5']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['omnet_2_5']['rss_loss'].rolling(10).mean(), 'C0', label='omnet_2_5')
    ax.plot(data_dict['omnet_5_5']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['omnet_5_5']['rss_loss'].rolling(10).mean(), 'C1', label='omnet_5_5')
    ax.plot(data_dict['omnet_10_10']['rss_loss'], 'C2', alpha=0.3)
    ax.plot(data_dict['omnet_10_10']['rss_loss'].rolling(10).mean(), 'C2', label='omnet_10_10')
    ax.plot(data_dict['omnet_30_10']['rss_loss'], 'C3', alpha=0.3)
    ax.plot(data_dict['omnet_30_10']['rss_loss'].rolling(10).mean(), 'C3', label='omnet_30_10')
    ax.plot(data_dict['omnet_10_30']['rss_loss'], 'C4', alpha=0.3)
    ax.plot(data_dict['omnet_10_30']['rss_loss'].rolling(10).mean(), 'C4', label='omnet_10_30')
    ax.grid(which='both', alpha=0.5)
    ax.set_yscale('log')
    ax.legend()
    return fig

def omnet_compare_bn_dp(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['omnet_10_30']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['omnet_10_30']['rss_loss'].rolling(10).mean(), 'C0', label='omnet_10_30')
    ax.plot(data_dict['omnet_10_30_bn']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['omnet_10_30_bn']['rss_loss'].rolling(10).mean(), 'C1', label='omnet_10_30_bn')
    ax.plot(data_dict['omnet_10_30_dp']['rss_loss'], 'C2', alpha=0.3)
    ax.plot(data_dict['omnet_10_30_dp']['rss_loss'].rolling(10).mean(), 'C2', label='omnet_10_30_dp')
    ax.plot(data_dict['omnet_10_30_bn_dp']['rss_loss'], 'C3', alpha=0.3)
    ax.plot(data_dict['omnet_10_30_bn_dp']['rss_loss'].rolling(10).mean(), 'C3', label='omnet_10_30_bn_dp')
    ax.grid(which='both', alpha=0.5)
    ax.set_yscale('log')
    ax.legend()
    return fig

def omnet_compare_polar(data_dict) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    ax.set_title('Validation Loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    # 10_10
    ax.plot(data_dict['omnet_10_30']['rss_loss'], 'C0', alpha=0.3)
    ax.plot(data_dict['omnet_10_30']['rss_loss'].rolling(10).mean(), 'C0', label='omnet_10_30')
    ax.plot(data_dict['omnet_10_30_polar']['rss_loss'], 'C1', alpha=0.3)
    ax.plot(data_dict['omnet_10_30_polar']['rss_loss'].rolling(10).mean(), 'C1', label='omnet_10_30_polar')
    ax.set_yscale('log')
    ax.grid(which='both', alpha=0.5)
    ax.legend()
    return fig

def omnet_prediction(model_path: str, gw_info: str, gw_ref: int, confidence: float=0.9):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,5))
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.set_title('RSSI Prediction')

    model = torch.load(model_path)
    x_mesh, y_mesh = np.mgrid[0:500:1, 0:500:1]
    x, y = x_mesh.ravel(), y_mesh.ravel()
    data_x = np.stack((x, y), axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss=True).detach()[:, gw_ref, :].cpu().numpy()
    z_conf = z[:, 1]
    z_val = z[:, 0]
    z_val[z_conf < confidence] = -1000

    z_val = z_val.reshape((500, 500)).transpose()
    z_conf = z_conf.reshape((500, 500)).transpose()

    pred_im = ax0.imshow(z_val)


    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Confidence prediction')
    pred_conf = ax1.imshow(z_conf)

    gw_data = DataReader.read_gw_info(gw_info)
    for gw_idx in gw_data:
        draw_gw(ax0, gw_idx)
        draw_gw(ax1, gw_idx)

    plt.tight_layout()
    plt.show()

def omnet_full_pred(model_path: str, obstacle_path: str, gw_info: str, confidence: float=0.6):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    z_im = np.ones((500, 500)) * -148
    conf_im = np.zeros((500, 500))

    model = torch.load(model_path)
    x_mesh, y_mesh = np.mgrid[0:500:1, 0:500:1]
    x, y = x_mesh.ravel(), y_mesh.ravel()
    data_x = np.stack((x, y), axis=1).astype(np.float32)
    z = model(torch.from_numpy(data_x), apply_ploss=True).detach()[:, 0, :].cpu().numpy()
    z_conf = z[:, 1]
    z_val = z[:, 0]
    z_val[z_conf < confidence] = -148

    z_im = z_im.ravel()
    conf_im = conf_im.ravel()
    z_im[z_val != -148] = z_val[z_val != -148]
    conf_im[z_conf > 0.1] = z_conf[z_conf>0.1]

    z = model(torch.from_numpy(data_x), apply_ploss=True).detach()[:, 1, :].cpu().numpy()
    z_conf = z[:, 1]
    z_val = z[:, 0]
    z_val[z_conf < confidence] = -148

    z_im[z_val != -148] = z_val[z_val != -148]
    conf_im[z_conf>0.1] = z_conf[z_conf>0.1]

    z_im = z_im.reshape((500, 500)).transpose()
    conf_im = conf_im.reshape((500, 500)).transpose()
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.set_title('RSSI Prediction')
    im = ax0.imshow(z_im, aspect='auto')

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title('Confidence prediction')
    ax1.imshow(conf_im, aspect='auto')

    gw_data = DataReader.read_gw_info(gw_info)
    for gw_idx in gw_data:
        draw_gw(ax0, gw_idx)
        draw_gw(ax1, gw_idx)

    with open(obstacle_path) as f:
        for line in f.readlines():
            obst_pts = [float(x) for x in line.strip().split(',')]
            draw_obstacle(ax0, obst_pts)
            draw_obstacle(ax1, obst_pts)

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical', label='dBm')

    return fig

def omnet_params_to_loss_ratio(data_dict: pd.DataFrame):
    data_keys = ['omnet_2_5', 'omnet_5_5', 'omnet_10_10', 'omnet_30_10', 'omnet_10_30']
    model_dict = {k: torch.load(f'models/{k}.pth') for k in data_keys}
    result_dict = {}
    for okey in data_keys:
        #mean_minimum = data_dict[okey]['rss_loss'].rolling(10).mean().min()
        mean_minimum = data_dict[okey]['rss_loss'].min()
        model_params = sum(p.numel() for p in model_dict[okey].parameters() if p.requires_grad)
        result_dict[okey] = {
            'mean_min': mean_minimum, 
            'params': model_params}
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    result = np.float32([[result_dict[k]['params'], result_dict[k]['mean_min']] for k in result_dict.keys()])
    x_label = [k + '\n' + str(result_dict[k]['params']) for k in data_keys]
    ax.bar(x_label, result[:, 1])
    #ax.scatter(x_label, result[:, 1])
    ax.set_xlabel('Model (no. of trainable parameters)')
    ax.set_ylabel('Minimum validation loss')
    ax.set_yscale('log')
    ax.set_title('Minimum loss over no. parameters')

def cross01_params_to_loss_ratio(data_dict: pd.DataFrame):
    data_keys = ['cross01_2_5', 'cross01_5_5', 'cross01_30_10', 'cross01_10_30']
    model_dict = {k: torch.load(f'models/{k}.pth') for k in data_keys}
    result_dict = {}
    for okey in data_keys:
        mean_minimum = data_dict[okey]['rss_loss'].rolling(10).mean().min()
        #mean_minimum = data_dict[okey]['rss_loss'].min()
        model_params = sum(p.numel() for p in model_dict[okey].parameters() if p.requires_grad)
        result_dict[okey] = {
            'mean_min': mean_minimum, 
            'params': model_params}
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    result = np.float32([[result_dict[k]['params'], result_dict[k]['mean_min']] for k in result_dict.keys()])
    x_label = [k + '\n' + str(result_dict[k]['params']) for k in data_keys]
    ax.bar(x_label, result[:, 1])
    #ax.scatter(x_label, result[:, 1])
    ax.set_xlabel('Model (no. of trainable parameters)')
    ax.set_ylabel('Minimum validation loss')
    ax.set_title('Minimum loss over no. parameters')

def omnet_gt_samples(data_path: str, obstacle_path: str, gw_info: str) -> plt.Figure:
    data_x, data_y, gw_pos = DataReader.read(data_path, FileType.OMNET, DataReader.read_gw_info(gw_info))
    
    data_y[data_y<-148] = -148

    sample_color = np.max(data_y, axis=1)    

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    
    ax.scatter(data_x[:, 0], data_x[:, 1], c=sample_color)

    for gw_idx in gw_pos:
        draw_gw(ax, gw_idx)

    with open(obstacle_path) as f:
        for line in f.readlines():
            obst_pts = [float(x) for x in line.strip().split(',')]
            draw_obstacle(ax, obst_pts)
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('omnet samples')
    plt.tight_layout()
    plt.show()
    return None
    
        







if __name__ == '__main__':
    eval_files = [f for f in os.listdir('eval/') if os.path.isfile(os.path.join('eval/', f))]
    data_dict = {k[:-4] : pd.read_csv('eval/' + k) for k in eval_files}

    fig = omnet_full_pred('models/omnet_30_10.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_30_10_full_pred.png')
    plt.close(fig)
    exit(0)


    fig = cross01_ground_truth('data/cross01_full.csv', 'data/cross01_obst.txt', 'data/cross01_gw.txt', 'recv_pwr0')
    plt.tight_layout()
    plt.savefig('images/cross01_full_gw0.png')
    plt.close(fig)
    fig = cross01_ground_truth('data/cross01_full.csv', 'data/cross01_obst.txt', 'data/cross01_gw.txt', 'recv_pwr1')
    plt.tight_layout()
    plt.savefig('images/cross01_full_gw1.png')
    plt.close(fig)
    fig = cross01_ground_truth('data/cross01_full.csv', 'data/cross01_obst.txt', 'data/cross01_gw.txt', 'recv_pwr2')
    plt.tight_layout()
    plt.savefig('images/cross01_full_gw2.png')
    plt.close(fig)
    fig = cross01_compare_loss(data_dict)
    plt.tight_layout()
    plt.savefig('images/cross01_loss_total.png')
    plt.close(fig)
    fig = cross01_compare_loss_no_mod(data_dict)
    plt.tight_layout()
    plt.savefig('images/cross01_loss_no_mod.png')
    plt.close(fig)
    fig = cross01_compare_bn_dp(data_dict)
    plt.tight_layout()
    plt.savefig('images/cross01_loss_bn_dp.png')
    plt.close(fig)
    fig = cross01_compare_polar(data_dict)
    plt.tight_layout()
    plt.savefig('images/cross01_loss_polar.png')
    plt.close(fig)
    fig = cross01_ground_truth_polar('data/cross01_full.csv', 'data/cross01_obst.txt', 'data/cross01_gw.txt', 0)
    plt.tight_layout()
    plt.savefig('images/cross01_polar_cartesian.png')
    plt.close(fig)
    fig = cross01_prediction_vs_gt('models/cross01_10_30.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_gt_pred.png')
    plt.close(fig)
    fig = cross01_prediction_vs_gt('models/cross01_10_30_polar.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_polar_gt_pred.png')
    plt.close(fig)
    fig = cross01_prediction_vs_gt('models/cross01_2_5.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_2_5_gt_pred.png')
    plt.close(fig)
    fig = cross01_prediction_vs_gt('models/cross01_5_5.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_5_5_gt_pred.png')
    plt.close(fig)
    fig = cross01_prediction_vs_gt('models/cross01_10_30_polar.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_polar_gt_pred.png')
    plt.close(fig)
    fig = cross01_prediction('models/cross01_10_30.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_pred.png')
    plt.close(fig)
    fig = cross01_prediction('models/cross01_10_30_polar.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_polar_pred.png')
    plt.close(fig)
    fig = cross01_prediction('models/cross01_2_5.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_2_5_pred.png')
    plt.close(fig)
    fig = cross01_prediction('models/cross01_5_5.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_5_5_pred.png')
    plt.close(fig)
    fig = cross01_prediction('models/cross01_10_30_polar.pth', 'data/cross01_full.csv', 'data/cross01_obst.txt', 
        'data/cross01_gw.txt', 0, 0.5)
    plt.tight_layout()
    plt.savefig('images/cross01_10_30_polar_pred.png')
    plt.close(fig)
    fig = omnet_compare_loss_no_mod(data_dict)
    plt.tight_layout()
    plt.savefig('images/omnet_loss_no_mod.png')
    plt.close(fig)
    fig = omnet_compare_bn_dp(data_dict)
    plt.tight_layout()
    plt.savefig('images/omnet_loss_bn_dp.png')
    plt.close(fig)
    fig = omnet_compare_polar(data_dict)
    plt.tight_layout()
    plt.savefig('images/omnet_loss_polar.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_10_30_polar.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_10_30_polar_full_pred.png')
    plt.close(fig)
    fig = omnet_params_to_loss_ratio(data_dict)
    plt.tight_layout()
    plt.savefig('images/omnet_params_to_loss_ratio.png')
    plt.close(fig)
    fig = cross01_params_to_loss_ratio(data_dict)
    plt.tight_layout()
    plt.savefig('images/cross01_params_to_loss_ratio.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_2_5.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_2_5_full_pred.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_5_5.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_5_5_full_pred.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_10_10.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_10_10_full_pred.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_10_30.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_10_30_full_pred.png')
    plt.close(fig)
    fig = omnet_full_pred('models/omnet_30_10.pth', 'data/omnet_wall_obst.txt', 'data/omnet_gw.txt', 0.1)
    plt.tight_layout()
    plt.savefig('images/omnet_30_10_full_pred.png')
    plt.close(fig)
    exit(0)

    
    exit(0)