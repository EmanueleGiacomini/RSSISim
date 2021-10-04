"""evaluator.py
"""

from lib.io_utils import DataReader, FileType, LoadFromFile
from lib.regressor import MultiGWRegressor, Trainer
from lib.plot_utils import *
from .launcher import bcolors

import argparse
import os
import torch


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(prog='rssisim_evaluator',
        description='Evaluator for RSSNet')
    parser.add_argument('--model', help='model file path. If train flag is enabled, the path is used to store the new '
        'model', type=str, required=False)
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
    parser.add_argument('--cuda', help='Enable CUDA routines for GPU training', type=bool, default=True, required=False)

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

    # Load pretrained model
    print(bcolors.OKCYAN + 'Evaluation mode' + bcolors.ENDC)
    model = torch.load(args.model)

    # Evaluation and plots
    # Build a trajectory
    # (21, 14)
    # (23, 27)
    # (31, 30)
    # (35, 31)
    # (41, 28)
    # (44, 23)
    # (51, 16)
    # (65, 21)
    # (70, 28)


    exit(0)