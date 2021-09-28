"""launcher.py
"""

from lib.io_utils import DataReader, FileType
from lib.regressor import MultiGWRegressor, Trainer
from lib.plot_utils import *

import argparse
import os
import torch

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
    parser.add_argument('--model', help='model file path. If train flag is enabled, the path is used to store the new '
                                        'model', type=str, required=True)
    parser.add_argument('--plot', help='plot flag. Plot the resulting map', required=False, action='store_true')
    parser.add_argument('--plot_samples', help='plot samples.', default=False, action='store_true',
                        required=False)
    parser.add_argument('--plot_training', help='Online plot regression boundaries during training', required=False,
                        action='store_true', default=False)
    parser.add_argument('--min_x', help='minimum x coordinate for plot area.', type=float, required=False, default=0)
    parser.add_argument('--min_y', help='minimum y coordinate for plot area.', type=float, required=False, default=0)
    parser.add_argument('--max_x', help='maximum x coordinate for plot area.', type=float, required=False, default=500)
    parser.add_argument('--max_y', help='maximum y coordinate for plot area.', type=float, required=False, default=500)
    parser.add_argument('--resolution', help='resolution of the plot grid', type=float, required=False, default=1)
    parser.add_argument('--data', help='dataset file path', type=str, required=True)
    parser.add_argument('--data_gw', help='gateway info file path', type=str, required=False)
    parser.add_argument('--data_type', help='dataset file type [rssisim | omnet]', type=str, required=False)

    args = parser.parse_args()

    # Parse data files ------------------------------------------
    data_fname = args.data
    if not os.path.isfile(data_fname):
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
    assert data_x is not None and data_y is not None and gw_pos is not []
    print(bcolors.OKGREEN + f'Read {data_x.shape[0]} samples.' + bcolors.ENDC)
    # Training / Evaluation section ----------------------------------------------------------
    if args.train:
        print(bcolors.OKCYAN + 'Training mode' + bcolors.ENDC)
        model = MultiGWRegressor(len(gw_pos), gw_pos, wl=args.wavelength)
        model = Trainer.train(data_x, data_y, model, optimizer=args.optimizer, epochs=args.epochs,
                              best_fit=True, early_stop=-1, lr=1e-3, plot_training=args.plot_training)
                              #best_fit=True, early_stop=-1, lr=1e-2, plot_training=args.plot_training)
        # Store model
        torch.save(model, args.model)
    else:
        print(bcolors.OKCYAN + 'Evaluation mode' + bcolors.ENDC)
        model = torch.load(args.model)
    # Plot section ----------------------------------------------------------------------------
    if args.plot:
        print(bcolors.OKGREEN + 'Plotting results' + bcolors.ENDC)
        fig, axs = build_generic_canvas(1, len(gw_pos), figsize=(len(gw_pos) * 5, 5))
        for i in range(len(gw_pos)):
            axs[i].set_title(f'Estimate for GW_{i}')
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            draw_model_prediction(fig, axs[i], model, i, (args.min_x, args.min_y), (args.max_x, args.max_y))
            if args.plot_samples:
                samples_scat = draw_samples(axs[i], data_x, data_y[:, i])
            gw_scat = draw_gw(axs[i], gw_pos[i])
            axs[i].set_xlim(args.min_x, args.max_x)
            axs[i].set_ylim(args.min_y, args.max_y)
        plt.tight_layout()
        plt.show()
    exit(0)
