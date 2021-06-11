"""data_plotter.py
"""

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_df = pd.read_csv('./data/data.csv')
    data_x = data_df['x'].to_numpy()
    data_y = data_df['y'].to_numpy()
    data_z0 = data_df['recv_pwr0'].to_numpy()
    data_z1 = data_df['recv_pwr1'].to_numpy()
    data_z2 = data_df['recv_pwr2'].to_numpy()

    data_z0[data_z0 < -50] = -50
    data_z1[data_z1 < -50] = -50
    data_z2[data_z2 < -50] = -50

    data_sqrt_len = int(np.sqrt(data_x.shape[0]))

    data_x = data_x.reshape((data_sqrt_len, data_sqrt_len))
    data_y = data_y.reshape((data_sqrt_len, data_sqrt_len))
    data_z0 = data_z0.reshape((data_sqrt_len, data_sqrt_len))
    data_z1 = data_z1.reshape((data_sqrt_len, data_sqrt_len))
    data_z2 = data_z2.reshape((data_sqrt_len, data_sqrt_len))
    fig, axs = plt.subplots(2, 2)
    axs[0,0].imshow(data_z0)
    axs[0,1].imshow(data_z1)
    axs[1,0].imshow(data_z2)
    axs[1,1].imshow(data_z0+data_z1+data_z2)
    plt.show()
    exit(0)
