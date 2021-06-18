"""launcher.py
"""

from lib.io_utils import DataReader, FileType
from lib.regressor import MultiGWRegressor, Trainer
from lib.plot_utils import *

import argparse
import os
import torch

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
        raise Exception(f'{data_fname} does not exists.')

    gw_pos = None
    if args.data_gw:
        data_gw_fname = args.data_gw
        if not os.path.isfile(data_fname):
            raise Exception(f'{data_fname} does not exists.')
        gw_pos = DataReader.read_gw_info(data_gw_fname)
    print('Loaded following GW info:')
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
        print('WARNING: no data file type passed. Using deduction heuristics')
        if 'omnet' in data_fname:
            data_ftype = FileType.OMNET
        else:
            data_ftype = FileType.RSSISIM
    print(f'Opening data file: {data_fname} as {data_ftype} type')
    data_x, data_y, gw_pos = DataReader.read(data_fname, data_ftype, gw_pos)
    assert data_x is not None and data_y is not None and gw_pos is not []
    print(f'Read {data_x.shape[0]} samples.')
    # Training / Evaluation section ----------------------------------------------------------
    if args.train:
        print('Training mode')
        model = MultiGWRegressor(len(gw_pos), gw_pos, wl=args.wavelength)
        model = Trainer.train(data_x, data_y, model, optimizer=args.optimizer, epochs=args.epochs,
                              best_fit=True, early_stop=-1, lr=1e-3)
        # Store model
        torch.save(model, args.model)
    else:
        print('Evaluation mode')
        model = torch.load(args.model)
    # Plot section ----------------------------------------------------------------------------
    if args.plot:
        print('Plotting results')
        fig, axs = build_generic_canvas(1, len(gw_pos), figsize=(len(gw_pos) * 10, 5))
        for i in range(len(gw_pos)):
            axs[i].set_title(f'Estimate for GW_{i}')
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')
            draw_model_prediction(fig, axs[i], model, i, (args.min_x, args.min_y), (args.max_x, args.max_y))
            draw_samples(axs[i], data_x)
        plt.tight_layout()
        plt.show()
    exit(0)
