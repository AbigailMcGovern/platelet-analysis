import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.measure import quantile_normalise_variables
import numpy as np


def timeplot_surface_comparison(df0, df1, y_col='dv', threshold=50, thresh_col='nb_density_15_pcntf'):
    df0['time (s)'] = df0['frame'] / 0.321764322705706
    df0['treatment'] = ['MIPS', ] * len(df0)
    df1['time (s)'] = df1['frame'] / 0.321764322705706
    df1['treatment'] = ['Saline', ] * len(df1)
    df = pd.concat([df0, df1])
    del df0
    del df1
    df = df.reset_index(drop=True)
    df = add_surface_variable(df, threshold=threshold, thresh_col=thresh_col)
    df['surface'] = df[thresh_col] < 50
    sdf = df[df['surface'] == True]
    cdf = df[df['surface'] == False]
    fig, axes = plt.subplots(2, 1, sharex=True)
    ax0, ax1 = axes.ravel()
    sns.lineplot(data=sdf, x='time (s)', y=y_col, hue='treatment', palette='rocket', lw=1, ax=ax0)
    #sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
    sns.lineplot(data=cdf, x='time (s)', y=y_col, hue='treatment', palette='rocket', lw=1, ax=ax1)
    #sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    plt.show()



def add_surface_variable(df, threshold=50, thresh_col='nb_density_15_pcntf'):
    df['surface'] = df[thresh_col] < 50
    return df


# Some box plots for growth vs consolidation (260 seconds, 180 seconds for letting go of mips surface platelets)
# total distance before release
# 

if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    mips_n = '211206_mips_df_220818.parquet'
    saline_n = '211206_saline_df_220827_amp0.parquet'
    mpath = os.path.join(d, mips_n)
    spath = os.path.join(d, saline_n)
    mdf = pd.read_parquet(mpath)
    sdf = pd.read_parquet(spath)
    mdf['nrterm'] = mdf['nrtracks'] - mdf['tracknr']
    sdf['nrterm'] = sdf['nrtracks'] - sdf['tracknr']
    #timeplot_surface_comparison(mdf, sdf, y_col='dv', threshold=50, thresh_col='nb_density_15_pcntf')
    #timeplot_surface_comparison(mdf, sdf, y_col='ca_corr', threshold=50, thresh_col='nb_density_15_pcntf')
    #timeplot_surface_comparison(mdf, sdf, y_col='nrterm', threshold=50, thresh_col='nb_density_15_pcntf')
    #timeplot_surface_comparison(mdf, sdf, y_col='nb_density_15', threshold=50, thresh_col='nb_density_15_pcntf')
    timeplot_surface_comparison(mdf, sdf, y_col='dvy', threshold=50, thresh_col='nb_density_15_pcntf')