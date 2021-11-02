"""data_reader.py
"""

from enum import Enum
import numpy as np
import pandas as pd


class FileType(Enum):
    RSSISIM = 0
    OMNET = 1


def _read_RSSISIM(fname: str, gw_pos: [np.float32] = None) -> (pd.DataFrame, [np.float32]):
    raw_df = pd.read_csv(fname)
    return raw_df, gw_pos


def _read_OMNET(fname: str, gw_pos: [np.float32] = None) -> (pd.DataFrame, [np.float32]):
    def search_keycols(df: pd.DataFrame) -> {str: str}:
        dkeys = df.keys()
        key_map = {}
        gw_ctr = 0
        for k in dkeys:
            if 'X Vector' in k:
                key_map['x'] = k
            if 'Y Vector' in k:
                key_map['y'] = k
        # Search for measurements
        n_gw = len(gw_pos)
        while gw_ctr < n_gw:
            for k in dkeys:
                if f'GW[' in k and 'Received RSSI' in k and 'Gateway' not in k:
                    key_map['recv_pwr' + str(gw_ctr)] = k
                    gw_ctr += 1
        return key_map

    raw_df = pd.read_csv(fname, sep=';')
    # Parse relevant keys from raw_df
    key_map = search_keycols(raw_df)
    proc_df = pd.DataFrame()
    gw_position_lst = []
    for k in key_map:
        proc_df[k] = raw_df[key_map[k]]
    # Parse GW infos from key_map
    if gw_pos is None:
        for k in key_map:
            df_key = key_map[k]
            if 'position' in df_key:
                subk = df_key.split('=')[-1].split(')')[0]
                gw_position_lst.append(np.float32(subk.split(',')))
    else:
        gw_position_lst = gw_pos
    return proc_df, gw_position_lst


class DataReader:
    _FTYPE_READ_CB_DICT = {FileType.RSSISIM: _read_RSSISIM, FileType.OMNET: _read_OMNET}

    @staticmethod
    def read(fname: str, ftype: FileType, gw_pos: [np.float32] = None) -> (np.float32, np.float32, [np.float32]):
        if ftype not in DataReader._FTYPE_READ_CB_DICT.keys():
            raise NotImplementedError(f'{ftype} is not a valid file type.')
        read_fn = DataReader._FTYPE_READ_CB_DICT[ftype]
        data_df, gw_pos_lst = read_fn(fname, gw_pos)
        data_x = data_df[['x', 'y']].to_numpy().astype(np.float32)
        gw_keys = []
        for k in data_df.keys():
            if 'recv_pwr' in k:
                gw_keys.append(k)
        data_y = data_df[gw_keys].to_numpy().astype(np.float32)
        return data_x, data_y, gw_pos_lst

    @staticmethod
    def read_gw_info(fname: str) -> [np.float32]:
        gw_pos = []
        with open(fname, 'r') as f:
            for line in f.readlines():
                gw_p = np.float32(line.strip().split(','))
                gw_pos.append(gw_p)
        return gw_pos

    @staticmethod
    def augmentDetection(data_y: np.array, detection_threshold: float) -> np.array:
        augm_y = np.zeros((data_y.shape[0], data_y.shape[1], 2))
        augm_y[:, :, 1] = data_y > detection_threshold        
        augm_y[:, :, 0] = data_y
        to_repl = augm_y[:, :, 0]
        to_repl[to_repl < detection_threshold] = detection_threshold
        augm_y[:, :, 0] = to_repl         
        return augm_y.astype(np.float32)


if __name__ == '__main__':
    print(f'Data_reader test')
    print('Reading Omnet data')
    fname_omnet = './data/omnet00.csv'
    data_x, data_y, gw_pos = DataReader.read(fname_omnet, FileType.OMNET)
    print(data_x, data_y, gw_pos)
    print('Reading RSSISIM data')
    fname_rssisim = './data/data.csv'
    fname_rssisim_gw = './data/data_gw.txt'
    gw_pos_rssisim = DataReader.read_gw_info(fname_rssisim_gw)
    data_x, data_y, gw_pos = DataReader.read(fname_rssisim, FileType.RSSISIM, gw_pos_rssisim)
    print(data_x, data_y, gw_pos)
