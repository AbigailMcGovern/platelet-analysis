import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plateletanalysis.variables.basic import time_seconds, cyl_bin, time_bin_1_2_5_10, time_minutes, add_terminating, get_treatment_name
from plateletanalysis.variables.transform import cylindrical_coordinates
from plateletanalysis.variables.measure import add_finite_diff_derivative
from plateletanalysis.analysis.peaks_analysis import smooth_vars, var_over_cylr, groupby_summary_data_mean, groupby_summary_data_counts
from plateletanalysis.variables.neighbours import local_contraction
from collections import defaultdict
from toolz import curry
import napari
from scipy.signal import find_peaks
import os


# ---------
# Functions
# ---------

def counts_and_density_over_time(df):
    gb = ['path', 'treatment', 'time (s)']
    plt.rcParams['svg.fonttype'] = 'none'
    dens_data = groupby_summary_data_mean(df, gb=gb, cols=['nb_density_15', ])
    dens_data = smooth_vars(dens_data, ['nb_density_15', ], gb='path')
    count_data = groupby_summary_data_counts(df, gb)
    count_data = smooth_vars(count_data, ['platelet count', ], gb='path')
    fig, axs = plt.subplots(1, 2)
    order = ['saline', 'bivalirudin', 'PAR4--']
    sns.lineplot(data=count_data, x='time (s)', y='platelet count', 
                 ax=axs[0], hue='treatment', palette='rocket', hue_order=order)
    sns.despine(ax=axs[0])
    sns.lineplot(data=dens_data, x='time (s)', y='nb_density_15', 
                 ax=axs[1], hue='treatment', palette='rocket', hue_order=order)
    sns.despine(ax=axs[1])
    plt.show()


def box_for_counts_comp(df):
    gb = ['path', 'treatment']
    plt.rcParams['svg.fonttype'] = 'none'
    count_data = groupby_summary_data_counts(df, gb)
    saline = count_data[count_data['treatment'] == 'saline']['platelet count'].values
    biva = count_data[count_data['treatment'] == 'bivalirudin']['platelet count'].values
    par4 = count_data[count_data['treatment'] == 'PAR4--']['platelet count'].values
    print('biva: count')
    print(stats.mannwhitneyu(saline, biva))
    print('par4: count')
    print(stats.mannwhitneyu(saline, par4))
    order = ['saline', 'bivalirudin', 'PAR4--']
    ax = sns.boxplot(data=count_data, x='treatment', y='platelet count', order=order, palette='rocket')
    sns.stripplot(data=count_data, x='treatment', y='platelet count', order=order, palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    plt.show()
    


def ind_counts_saline_biva(df):
    gb = ['path', 'treatment', 'time (s)']
    plt.rcParams['svg.fonttype'] = 'none'
    count_data = groupby_summary_data_counts(df, gb)
    count_data = smooth_vars(count_data, ['platelet count', ], gb='path')
    fig, axs = plt.subplots(1, 2, sharey=True)
    biva = count_data[count_data['treatment'] == 'bivalirudin']
    veh = count_data[count_data['treatment'] == 'saline']
    sns.lineplot(data=veh, x='time (s)', y='platelet count', 
                 ax=axs[0], hue='path', palette='rocket')
    sns.despine(ax=axs[0])
    sns.lineplot(data=biva, x='time (s)', y='platelet count', 
                 ax=axs[1], hue='path', palette='rocket')
    sns.despine(ax=axs[1])
    plt.show()



# -------
# Execute
# -------
if __name__ == '__main__':
    #do = 'counts_and_density'
    do = 'ind_counts'

    df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
    df = df[df['nrtracks'] > 1]

    if do == 'counts_and_density':
        counts_and_density_over_time(df)
        #box_for_counts_comp(df)


    elif do == 'ind_counts':
        counts_and_density_over_time(df)
        ind_counts_saline_biva(df)