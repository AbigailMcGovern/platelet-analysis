import pandas as pd
import os
import numpy as np

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
biva = os.path.join(d, '211206_biva_df.parquet')
ctrl = os.path.join(d, '211206_ctrl_df.parquet')

bdf = pd.read_parquet(biva)
cdf = pd.read_parquet(ctrl)

bdf.columns.values

for c in bdf.columns.values:
    print(c, type(bdf[c].values[0]))


