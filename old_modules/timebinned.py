from ripser import ripser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from scipy import stats
from plateletanalysis.variables.measure import quantile_normalise_variables_frame



# '211206_veh-mips_df_220831.parquet'
def variable_versus_barcode_lifespan(df, var='platelet count', time_bins=((0, 60), (60, 180), (180, 600)), units='%'):
    tx_col = 'treatment'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    sns.set_style("ticks")
    fig, axes = plt.subplots(len(time_bins), 2, sharey=True, sharex='col')
    for i, bin in enumerate(time_bins):
        t_min, t_max = bin
        data = df[(df[t_col] > t_min) & (df[t_col] < t_max)]
        data = injury_averages(data, units)
        if len(time_bins) == 1:
            ax0 = axes[0]
            ax1 = axes[1]
        else:
            ax0 = axes[i, 0]
            ax1 = axes[i, 1]
        slope, intercept = _stats_with_print(l_col, var, f'time bin: {bin} seconds | {var} vs lifespan', data)
        sns.scatterplot(x=l_col, y=var, hue=tx_col, data=data, ax=ax0) # palette="husl"
        sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
        ax0.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
        slope, intercept = _stats_with_print(o_col, var, f'time bin: {bin} seconds | {var} vs outlierness', data)
        sns.scatterplot(x=o_col, y=var, hue=tx_col, data=data, ax=ax1)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    plt.show()



def count_versus_barcode_lifespan(df, time_bins=((0, 60), (60, 180), (180, 600)), units='%'):
    tx_col = 'treatment'
    p_col = 'path' 
    f_col = 'frame'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    turn_col = 'turnover (%)'
    c_col = 'platelet count'
    sns.set_style("ticks")
    fig, axes = plt.subplots(len(time_bins), 2, sharey=True, sharex='col')
    for i, bin in enumerate(time_bins):
        t_min, t_max = bin
        data = df[(df[t_col] > t_min) & (df[t_col] < t_max)]
        data = injury_averages(data, units)
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]
        slope, intercept = _stats_with_print(l_col, c_col, f'time bin: {bin} seconds | turnover vs lifespan', data)
        sns.scatterplot(x=l_col, y=c_col, hue=tx_col, data=data, ax=ax0) # palette="husl"
        sns.move_legend(ax0, "upper left", bbox_to_anchor=(1, 1))
        ax0.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
        slope, intercept = _stats_with_print(l_col, c_col, f'time bin: {bin} seconds | turnover vs outlierness', data)
        sns.scatterplot(x=o_col, y=c_col, hue=tx_col, data=data, ax=ax1)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
        ax1.axline((0, intercept), (1, intercept + slope), ls='--', c='black')
    plt.show()


def injury_averages(df, units='%'):
    tx_col = 'treatment'
    p_col = 'path' 
    f_col = 'frame'
    t_col = 'time (s)'
    l_col = f'radius {units}'
    o_col = 'Standard deviations from mean'
    turn_col = 'turnover (%)'
    c_col = 'platelet count'
    dv_col = 'dv (um/s)'
    ca_col = 'corrected calcium'
    dens_col = 'density (platelets/um^2)'
    paths = pd.unique(df[p_col])
    df_av = {
        tx_col : [df[df[p_col] == p][tx_col].values[0] for p in paths], 
        p_col : [df[df[p_col] == p][p_col].values[0] for p in paths], 
        f_col : [df[df[p_col] == p][f_col].values[0] for p in paths], 
        t_col : [df[df[p_col] == p][t_col].mean() for p in paths], 
        l_col : [df[df[p_col] == p][l_col].mean() for p in paths], 
        o_col : [df[df[p_col] == p][o_col].mean() for p in paths],
        turn_col : [df[df[p_col] == p][turn_col].mean() for p in paths],  
        c_col : [df[df[p_col] == p][c_col].mean() for p in paths], 
        dv_col : [df[df[p_col] == p][dv_col].mean() for p in paths], 
        ca_col : [df[df[p_col] == p][ca_col].mean() for p in paths], 
        dens_col : [df[df[p_col] == p][dens_col].mean() for p in paths], 
    }
    df_av = pd.DataFrame(df_av)
    return df_av




def _stats_with_print(x_col, y_col, desc, data):
    x = data[x_col]
    y = data[y_col]
    print(f'Pearson R for {desc}')
    r, p = stats.pearsonr(x, y)
    print(f'r = {r}, p = {p}')
    print(f'Linear regression for {desc}')
    slope, intercept, r, p, se = stats.linregress(x, y)
    rsq = r ** 2
    print(f'slope = {slope}, intercept = {intercept}, r = {r}, r squared = {rsq}, se = {se}, p = {p}')
    return slope, intercept
