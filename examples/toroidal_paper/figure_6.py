import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from plateletanalysis.variables.basic import get_treatment_name, time_seconds, inside_injury_var, cyl_bin, time_bin_1_2_5_10
from plateletanalysis.variables.transform import cylindrical_coordinates
from plateletanalysis.analysis.peaks_analysis import smooth_vars, groupby_summary_data_mean

# these functions were added to the peaks_analysis module

def plot_var_over_cylr(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    cols = [col, ]
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=['path', 'time_bin'], t='cylr_bin', w=4)
    print(data.head)
    print(len(data))
    print(len(pd.unique(data.path)))
    data = data.dropna()
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.lineplot(data=data, y=col, x='cylr_bin', palette='rocket', hue='time_bin')
    ax.axline((37.5, 0), (37.5, 0.001), color='grey')
    ax.set_xlim(0, 75)
    #ax.set_ylim(-1, 0.5)
    #print(stats.linregress(data[dens].values, data[psel].values))
    sns.despine()
    plt.show()


def plot_var_over_cylr_ind(df, col):
    gb = ['path', 'cylr_bin']
    cols = [col, ]
    df = df[df['time_bin'] == '0-60 s']
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=['path', ], t='cylr_bin', w=8)
    data = data.dropna()
    print(data.head)
    print(len(data))
    print(len(pd.unique(data.path)))
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.lineplot(data=data, y=col, x='cylr_bin', palette='rocket', hue='path')
    ax.axline((37.5, 0), (37.5, 0.001), color='grey')
    ax.set_xlim(0, 75)
    #ax.set_ylim(-1, 0.5)
    #print(stats.linregress(data[dens].values, data[psel].values))
    sns.despine()
    plt.show()

# probably want to record the distance at which the line crosses 0

def value_based_box(df, col):
    gb = ['path', 'cylr_bin']
    cols = [col, ]
    df = df[df['time_bin'] == '0-60 s']
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(df, vars=cols, gb=['path', ], t='cylr_bin', w=8)
    data = mark_as_bookends(data, gb, col)
    data = data[data['bookends'] == True]
    data = data.groupby(['path', ])['cylr_bin'].mean().reset_index()
    data = data.dropna()
    print(data)
    print(len(data))
    print(len(pd.unique(data.path)))
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.boxplot(data=data, y='cylr_bin', palette='rocket')
    sns.stripplot(data=data, y='cylr_bin', ax=ax, palette='rocket', edgecolor='white', linewidth=0.3, jitter=True, size=5,)
    sns.despine()
    plt.show()


def mark_as_bookends(data, gb, col):
    data = data.sort_values('cylr_bin')
    for k, grp in data.groupby(gb):
        prev = None
        vals = []
        idx = grp.index.values
        for i, v in enumerate(grp[col].values):
            if prev is not None and 0 > prev and 0 < v:
                vals.append(True)
            else:
                vals.append(False)
            prev = v
        data.loc[idx, 'bookends'] = vals
    return data


def value_based_box_mins(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    cols = [col, ]
    df = df[(df['time_bin'] == '0-60 s') | (df['time_bin'] == '60-120 s')]
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=['path', 'time_bin'], t='cylr_bin', w=8)
    data = mark_as_bookends(data, gb, col)
    data = data[data['bookends'] == True]
    data = data.groupby(['path', 'time_bin'])['cylr_bin'].mean().reset_index()
    data = data.dropna()
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.boxplot(data=data, y='cylr_bin', x='time_bin', palette='rocket')
    sns.stripplot(data=data, y='cylr_bin',  x='time_bin', ax=ax, palette='rocket', edgecolor='white', linewidth=0.3, jitter=True, size=5,)
    sns.despine()
    plt.show()

    


if __name__ == '__main__':
    p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
    df = pd.read_parquet(p)
    df = df[df['nrtracks'] > 1]
    if 'time (s)' not in df.columns.values:
        df = time_seconds(df)
    if 'cyl_radial' not in df.columns.values:
        df = cylindrical_coordinates(df)
    df['cylr_bin'] = df['cyl_radial'].apply(cyl_bin)
    df['time_bin'] = df['time (s)'].apply(time_bin_1_2_5_10)
    #plot_var_over_cylr(df, 'dvz')
    #plot_var_over_cylr_ind(df, 'dvz')
    #plot_var_over_cylr(df, 'cont_p')
    #value_based_box(df, 'dvz')
    #value_based_box_mins(df, 'dvz')
    plot_var_over_cylr(df, 'ca_corr')
