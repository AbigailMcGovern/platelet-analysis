import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plateletanalysis.analysis.plots import pal1

save_path = '/Users/abigailmcgovern/Data/platelet-analysis/TDA/2305_donutness/230516_saline_DMSO-SQ_donutness.csv'
data = pd.read_csv(save_path)



def donutness_plot(df, y='donutness', x='time (s)'):
    df = df.sort_values(x)
    _add_rolled(y, df)
    sns.lineplot(data=data, x=x, y=y, hue='path')#palette=pal1, hue='treatment')
    plt.show()

def _add_rolled(col, df):
    for k, grp in df.groupby('path'):
        idxs = grp.index.values
        roll = grp[col].rolling(window=5,center=True).mean()
        df.loc[idxs, col] = roll

data = data[data['treatment'] == 'saline']
donutness_plot(data)