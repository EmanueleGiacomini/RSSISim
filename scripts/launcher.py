"""launcher.py
"""

from lib.io_utils import DataReader, FileType, LoadFromFile
from lib.regressor import MultiGWRegressor, Trainer
from lib.plot_utils import *

import argparse
import os
import torch
import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='rssisim_launcher',
                                     description='Launcher for RSSI model regressor')
    parser.add_argument('--train', help='training flag. Enables training routine', action='store_true')
    parser.add_argument('--optimizer', help='optimizer to use during training phase', default='adam', required=False)
    parser.add_argument('--wavelength', help='wavelength of technology used', type=float, default=0.34538302,
                        required=False)
    parser.add_argument('--epochs', help='number of epochs required to train the model', type=int, default=40000,
                        required=False)
    parser.add_argument('--lr', help='learning rate of the model', type=float, default=1e-3, required=False)
    parser.add_argument('--model', help='model file path. If train flag is enabled, the path is used to store the new '
                                        'model', type=str, required=False)
    parser.add_argument('--plot', help='plot flag. Plot the resulting map', required=False, action='store_true')
    parser.add_argument('--plot_samples', help='plot samples.', default=False, action='store_true',
                        required=False)
    parser.add_argument('--plot_training', help='Online plot regression boundaries during training', required=False,
                        action='store_true', default=False)
    parser.add_argument('--plot_obst', help='plot obstacles.', default=False, action='store_true',
                        required=False)
    parser.add_argument('--min_x', help='minimum x coordinate for plot area.', type=float, required=False, default=0)
    parser.add_argument('--min_y', help='minimum y coordinate for plot area.', type=float, required=False, default=0)
    parser.add_argument('--max_x', help='maximum x coordinate for plot area.', type=float, required=False, default=100)
    parser.add_argument('--max_y', help='maximum y coordinate for plot area.', type=float, required=False, default=100)
    parser.add_argument('--resolution', help='resolution of the plot grid', type=float, required=False, default=1)
    parser.add_argument('--data', help='dataset file path', type=str, required=False)
    parser.add_argument('--data_gw', help='gateway info file path', type=str, required=False)
    parser.add_argument('--data_obst', help='obstacles info file path', type=str, required=False)
    parser.add_argument('--data_type', help='dataset file type [rssisim | omnet]', type=str, required=False)
    parser.add_argument("--config", help="configuration file for the current execution", type=open, required=False, default=None,
        action=LoadFromFile)

    parser.add_argument('--h_dim', help="dimension of latent space for neural network (height of layer)", type=int, default=10, required=False)
    parser.add_argument('--n_layers', help="number of layers for the neural network (depth of network)", type=int, default=5, required=False)
    parser.add_argument('--polar', help="Transform model's feature into polar coordinates", required=False, action='store_true')
    parser.add_argument('--batch', help="Size of training batches", type=int, default=64, required=False)
    parser.add_argument('--tx_pwr', help="Tx power of the moving beacon in Watts", type=float, default=1, required=False)
    parser.add_argument('--detection_flag', help='Enable detection boolean estimation', required=False, action='store_true')
    parser.add_argument('--detection_threshold', help='Minimum value in dBm for detection. Readings under this value are considered undetected',
        type=float, default=-140, required=False)
    parser.add_argument('--cuda', help='Enable CUDA routines for GPU training', type=bool, default=True, required=False)
    parser.add_argument('--cw1', help='Cost weight for regression on available measurements', type=float, default=10., required=False)
    parser.add_argument('--cw2', help='Cost weight for regression on availability', type=float, default=1., required=False)
    parser.add_argument('--batch_norm', help='Add Batch Normalization layers on the model', action='store_true', required=False)
    parser.add_argument('--dropout', help='Add a Dropout layer on the model', action='store_true', required=False)
    parser.add_argument('--dropout_rate', help='Dropout rate. Requires dropout parameter', type=float, default=0.2, required=False)


    args = parser.parse_args()

    # Print node parameters
    args_dict = vars(args)
    print(bcolors.OKCYAN + 'Node Parameters:', bcolors.ENDC)
    for k in args_dict.keys():
        print(bcolors.WARNING, k, ':\t', bcolors.ENDC, args_dict[k])

    # Parse data files ------------------------------------------
    data_fname = args.data
    if not data_fname or not os.path.isfile(data_fname):
        raise Exception(bcolors.FAIL + f'{data_fname} does not exists.' + bcolors.ENDC)

    gw_pos = None
    if args.data_gw:
        data_gw_fname = args.data_gw
        if not os.path.isfile(data_fname):
            raise Exception(f'{data_fname} does not exists.')
        gw_pos = DataReader.read_gw_info(data_gw_fname)
    print(bcolors.OKGREEN + 'Loaded following GW info:' + bcolors.ENDC)
    for i, p in enumerate(gw_pos):
        print(f'GW_{i}={p}')

    data_ftype = 0
    if args.data_type:
        if args.data_type == 'rssisim':
            data_ftype = FileType.RSSISIM
        elif args.data_type == 'omnet':
            data_ftype = FileType.OMNET
        else:
            raise Exception('Unknown data format for data file: ' + data_fname)
    else:
        print(bcolors.WARNING + 'WARNING: no data file type passed. Using deduction heuristics' + bcolors.ENDC)
        if 'omnet' in data_fname:
            data_ftype = FileType.OMNET
        else:
            data_ftype = FileType.RSSISIM
    print(bcolors.OKGREEN + f'Opening data file: {data_fname} as {data_ftype} type' + bcolors.ENDC)
    data_x, data_y, gw_pos = DataReader.read(data_fname, data_ftype, gw_pos)
    # Augment data_y if detection_flag is set
    if args.detection_flag:
        data_y = DataReader.augmentDetection(data_y, args.detection_threshold)

    assert data_x is not None and data_y is not None and gw_pos is not []
    print(bcolors.OKGREEN + f'Read {data_x.shape[0]} samples.' + bcolors.ENDC)
    # Training / Evaluation section ----------------------------------------------------------
    if args.train:
        # Train/Test split

        print(bcolors.OKCYAN + 'Training mode' + bcolors.ENDC)
        model = MultiGWRegressor(len(gw_pos), gw_pos, wl=args.wavelength, n_layers=args.n_layers, h_dim=args.h_dim, tx_pwr=args.tx_pwr,
            detection_estimation=args.detection_flag, dropout=args.dropout, dropout_rate=args.dropout_rate, batch_norm=args.batch_norm)
        print(sum([p.numel() for p in model.parameters()]) / 3)
        model, model_df = Trainer.train(data_x, data_y, model, optimizer=args.optimizer, epochs=args.epochs,
                              best_fit=True, early_stop=-1, lr=args.lr, plot_training=args.plot_training,
                              batch_size=args.batch, train_test_split=args.train_split, detection_flag=args.detection_flag,
                              cuda_en=args.cuda, cost_weights=(args.cw1, args.cw2))
                              #best_fit=True, early_stop=-1, lr=1e-2, plot_training=args.plot_training)
        # Store model
        torch.save(model, args.model)
        # Store dataframe
        model_name = args.model[args.model.find('/'):args.model.find('.pth')]
        model_df.to_csv('./eval/' + model_name + '.csv')
    else:
        print(bcolors.OKCYAN + 'Evaluation mode' + bcolors.ENDC)
        model = torch.load(args.model)
    # Plot section ----------------------------------------------------------------------------
    if args.plot:
        print(bcolors.OKGREEN + 'Plotting results' + bcolors.ENDC)
        model_name = args.model[args.model.find('/'):args.model.find('.pth')]
        fig, axs = build_generic_canvas(1, len(gw_pos), figsize=(len(gw_pos) * 5, 5))
        for i in range(len(gw_pos)):
            axs[i].set_title(f'GW_{i} estimate')
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            draw_model_prediction(fig, axs[i], model, i, (args.min_x, args.min_y), (args.max_x, args.max_y), 
                detection_flag=args.detection_flag, detection_threshold=args.detection_threshold)
            if args.plot_samples:
                samples_scat = draw_samples(axs[i], data_x, data_y[:, i], 
                    detection_flag=args.detection_flag, detection_threshold=args.detection_threshold)
            gw_scat = draw_gw(axs[i], gw_pos[i])
            if args.plot_obst and args.data_obst:
                with open(args.data_obst) as f:
                    for line in f.readlines():
                        obst_pts = [float(x) for x in line.strip().split(',')]
                        draw_obstacle(axs[i], obst_pts)
            axs[i].set_xlim(args.min_x, args.max_x)
            axs[i].set_ylim(args.min_y, args.max_y)
        
        plt.tight_layout()
        plt.show()
        #plt.savefig('./images/' + model_name + '_intensity.jpg')
        # Training history
        """
        if args.train:
            plt.clf()
            fig, axs = build_generic_canvas(1, 1, figsize=(10, 10))
            axs.set_title(f'{model_name} training history')
            axs.set_xlabel('epochs')
            axs.set_ylabel('loss')
            axs.plot(train_val_loss, 'C1', label='val loss')
            axs.legend()
            plt.tight_layout()
            #plt.savefig('./images/' + model_name + '_train_hist.jpg')
        """

    exit(0)
