# %load_ext autoreload
# %autoreload 2

from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
saline_n = '211206_saline_df_220614-amp0.parquet'
path = os.path.join(d, saline_n)
df = pd.read_parquet(path)

# Clean the data
df = df[df['nrtracks'] > 50] # tracked for longer than 30s

df50 = df[df['nd15_percentile'] > 40]

# 31.08 seconds
df_f10 = df50[df50['frame'] == 10]
#mean = df_f10
paths = pd.unique(df_f10['path'])
for p in paths:
    sml_df = df_f10[df_f10['path'] == p]
    data = sml_df[['x_s', 'ys']].values

    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.show()

    dgms = ripser(data)['dgms']
    plot_diagrams(dgms, show=True)
    plt.show()


df60 = df[df['nd15_percentile'] > 80]
df_f10 = df60[df60['frame'] == 10]
data = df_f10[['x_s', 'ys']].values
df_f10t11 = df60[(df60['frame'] >= 10) & (df60['frame'] <= 11)]
data = df_f10t11[['x_s', 'ys']].values

plt.scatter(data[:, 0], data[:, 1], s=1)
plt.show()
dgms = ripser(data)['dgms']
plot_diagrams(dgms, show=True)