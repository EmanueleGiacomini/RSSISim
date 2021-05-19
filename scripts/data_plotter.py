"""data_plotter.py
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_df = pd.read_csv('./data/data.csv')
    data_x = data_df['x'].to_numpy()
    data_y = data_df['y'].to_numpy()
    data_z = data_df['recv_pwr'].to_numpy()

    data_sqrt_len = int(np.sqrt(data_x.shape[0]))

    data_x = data_x.reshape((data_sqrt_len, data_sqrt_len))
    data_y = data_y.reshape((data_sqrt_len, data_sqrt_len))
    data_z = data_z.reshape((data_sqrt_len, data_sqrt_len))
    print(data_z.shape)
    plt.contourf(data_x, data_y, data_z, levels=400)
    plt.show()
    exit(0)