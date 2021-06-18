"""omnet_preproc.py
"""

import pandas as pd
import numpy as np

OMNET_KEY_IDX = {'x': 6, 'y': 7, 'recv_pwr0': 2, 'recv_pwr1': 3}

"""
def parse_omnet(fname: str):
    raw_df = pd.read_csv(fname, sep=';')
    raw_keys = raw_df.keys()
    data_df = pd.DataFrame(columns=list(OMNET_KEY_IDX.keys()))
    for k in OMNET_KEY_IDX.keys():
        data_df[k] = raw_df[raw_keys[OMNET_KEY_IDX[k]]]
    return data_df
"""


def parse_omnet(fname: str, gw_pos: [np.array]=None) -> (pd.DataFrame, [np.array]):
    def search_keycols(df: pd.DataFrame) -> {str: str}:
        dkeys = df.keys()
        key_map = {}
        gw_ctr = 0
        for k in dkeys:
            if 'GW[' in k and 'Received RSSI' in k and 'Gateway' not in k:
                key_map['recv_pwr' + str(gw_ctr)] = k
                gw_ctr += 1
            if 'X Vector' in k:
                key_map['x'] = k
            if 'Y Vector' in k:
                key_map['y'] = k
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


if __name__ == '__main__':
    data_fname = './data/test-0617-wall-final.csv'
    print('Omnet_preproc test')
    print(f'Opening file:={data_fname}')
    data_df, gw_positions = parse_omnet(data_fname, gw_pos=[np.float32((240, 240)), np.float32((200, 140))])
    print(data_df.head())
    for i, gw_pos in enumerate(gw_positions):
        print(f'GW_{i}={gw_pos}')

    #print(data_df['recv_pwr1'].to_numpy().astype(np.float32))
    exit(0)
