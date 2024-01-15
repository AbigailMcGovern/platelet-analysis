import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plateletanalysis.variables.basic import time_seconds, cyl_bin, time_bin_1_2_5_10, time_minutes, add_terminating, get_treatment_name
from plateletanalysis.variables.transform import cylindrical_coordinates
from plateletanalysis.variables.measure import add_finite_diff_derivative
from plateletanalysis.analysis.peaks_analysis import smooth_vars, var_over_cylr, groupby_summary_data_mean
from plateletanalysis.variables.neighbours import local_contraction
from collections import defaultdict
from toolz import curry
import napari
from scipy.signal import find_peaks
import os


# ---------
# Functions
# ---------

# LOAD data
# ---------

def load_data_all():
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/'
    ns = ['211206_saline_df_toroidal-coords-1.parquet', '211206_ctrl_df_toroidal-coords.parquet', 
          '211206_cang_df_toroidal-coords.parquet', '211206_biva_df_toroidal-coords.parquet', 
          '211206_par4--_df_toroidal-coords.parquet']
    ps = [os.path.join(d, n) for n in ns]
    df = [pd.read_parquet(p) for p in ps]
    df = pd.concat(df).reset_index(drop=True)
    df = time_minutes(df)
    if 'time (s)' not in df.columns.values:
        df = time_seconds(df)
    if 'cyl_radial' not in df.columns.values:
        df = cylindrical_coordinates(df)
    df['cylr_bin'] = df['cyl_radial'].apply(cyl_bin)
    df['time_bin'] = df['time (s)'].apply(time_bin_1_2_5_10)
    df['treatment'] = df['path'].apply(get_treatment_name)
    df['tor_theta_diff'] = - df['tor_theta_diff'] * 0.32
    return df


# TOROIDAL MOVEMENT & LOCAL CONTRACTION  - cang, biva, par4 vs veh/ctl
# --------------------------------------------------------------------
def var_over_time_spec(df, t, var):
    plt.rcParams['svg.fonttype'] = 'none'
    #df = df[df['time (s)'] < 600]
    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [var, ])
    data = smooth_vars(data, [var, ], gb='path')
    data['treatment'] = data['path'].apply(get_treatment_name)
    df_biva = data[(data['treatment'] == 'saline') | (data['treatment'] == 'bivalirudin')]
    df_cang = data[(data['treatment'] == 'saline') | (data['treatment'] == 'cangrelor')]
    df_par4 = data[(data['treatment'] == 'saline') | (data['treatment'] == 'PAR4--')]
    fig, axs = plt.subplots(1, 3, sharey=True)
    sns.lineplot(data=df_biva, x=t, y=var, hue='treatment', ax=axs[0], palette='rocket')
    axs[0].set_title('Bivalirudin')
    sns.lineplot(data=df_par4, x=t, y=var, hue='treatment', ax=axs[1], palette='rocket')
    axs[1].set_title('PAR4 -/-')
    sns.lineplot(data=df_cang, x=t, y=var, hue='treatment', ax=axs[2], palette='rocket')
    axs[2].set_title('Cangrelor')
    sns.despine()
    plt.show()


def boxplots_for_min_1(df, var):
    plt.rcParams['svg.fonttype'] = 'none'
    data = groupby_summary_data_mean(df, ['path', 'minute'], [var, ])
    data['treatment'] = data['path'].apply(get_treatment_name)
    data_1_min = data[data['minute'] == 1]
    order = ['saline', 'bivalirudin', 'PAR4--', 'cangrelor']
    saline = data_1_min[data_1_min['treatment'] == 'saline'][var].values
    biva = data_1_min[data_1_min['treatment'] == 'bivalirudin'][var].values
    par4 = data_1_min[data_1_min['treatment'] == 'PAR4--'][var].values
    cang = data_1_min[data_1_min['treatment'] == 'cangrelor'][var].values
    print('biva: ', var)
    print(stats.mannwhitneyu(saline, biva))
    print('par4: ', var)
    print(stats.mannwhitneyu(saline, par4))
    print('cang: ', var)
    print(stats.mannwhitneyu(saline, cang))
    ax = sns.boxplot(data=data_1_min, x='treatment', order=order, y=var, palette='rocket')
    sns.stripplot(data=data_1_min, x='treatment', order=order, y=var, palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    plt.show()


# DONUTNESS - cang vs saline
# --------------------------

def var_over_time(df, t, var):
    plt.rcParams['svg.fonttype'] = 'none'
    #df = df[df['time (s)'] < 600]
    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [var, ])
    data = smooth_vars(data, [var, ], gb='path')
    data['treatment'] = data['path'].apply(get_treatment_name)
    sns.lineplot(data=data, x=t, y=var, hue='treatment')
    sns.despine()
    plt.show()


def donut_boxplot(data):
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.boxplot(data=data, x='treatment', y='donutness magnetude', palette='rocket')
    sns.stripplot(data=data, x='treatment', y='donutness magnetude', palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    plt.show()


# DENS over CYLR - cang
# ---------------------

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
    sns.despine()
    plt.show()

def plot_var_over_cylr_ind(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    cols = [col, ]
    data = groupby_summary_data_mean(df, gb, cols)
    data = smooth_vars(data, vars=cols, gb=['path', 'time_bin'], t='cylr_bin', w=4)
    data['treatment'] = data['path'].apply(get_treatment_name)
    print(data.head)
    print(len(data))
    print(len(pd.unique(data.path)))
    data = data.dropna()
    data = data[data['time_bin'] == '0-60 s']
    plt.rcParams['svg.fonttype'] = 'none'
    ax = sns.lineplot(data=data, y=col, x='cylr_bin', palette='rocket', hue='path')
    ax.axline((37.5, 0), (37.5, 0.001), color='grey')
    sns.despine()
    plt.show()


def path_dist_at_max(data, gb, col, dist='cylr_bin'):
    # return df of path and max
    out = defaultdict(list)
    for k, grp in data.groupby(gb):
        for i, c in enumerate(gb):
            out[c].append(k[i])
        midx = np.argmax(grp[col].values)
        m = grp[col].values[midx]
        md = grp[dist].values[midx]
        out[col].append(m)
        out[dist].append(md)
    out = pd.DataFrame(out)
    return out


def plot_peak_cylr_min1_box(df, col):
    gb = ['path', 'time_bin', 'cylr_bin']
    cols = [col, ]
    data = groupby_summary_data_mean(df, gb, cols)
    data_max = path_dist_at_max(data, ['path', 'time_bin'], 'nb_density_15', dist='cylr_bin')
    data_max['treatment'] = data['path'].apply(get_treatment_name)
    data_1_min = data_max[data_max['time_bin'] == '0-60 s']
    ax = sns.boxplot(data=data_1_min, y='cylr_bin', palette='rocket')
    sns.stripplot(data=data_1_min, y='cylr_bin', palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    plt.show()
    



# -------
# Execute
# -------
if __name__ == '__main__':
    #do = 'tor_movement' # DONE
    #do = 'local_cont'
    #do = 'donutness_cang'
    #do = 'dens_cylr_cang'
    do = 'move_stats'


    # TOROIDAL MOVEMENT - cang, biva, par4 vs veh/ctl
    if do == 'tor_movement':
        #df = load_data_all()
        #sp = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet'
        #df.to_parquet(sp)
        df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        var_over_time_spec(df, 'time (s)', 'tor_theta_diff')
        boxplots_for_min_1(df, 'tor_theta_diff')


    # LOCAL CONTRACTION -  cang, biva, par4 vs veh/ctl
    elif do == 'local_cont':
        df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        df = local_contraction(df)
        boxplots_for_min_1(df, 'nb_cont_15')
        boxplots_for_min_1(df, 'dv')


    # DONUTNESS - cang (n = 12) vs saline (n = 15)
    elif do == 'donutness_cang':
        # data for box plots
        p = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231012_mega_summary_data.csv'
        sum_data = pd.read_csv(p)
        sum_data['treatment'] = sum_data['path'].apply(get_treatment_name)
        # create box
        sum_data = sum_data[(sum_data['treatment'] == 'saline') | (sum_data['treatment'] == 'cangrelor')]
        donut_boxplot(sum_data)
        # data for time plots
        d = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
        ns = ['cangrelor_donut_data_scaled_sn200_n100_c50.csv', 'saline_donut_data_scaled_sn200_n100_c50_gt1tks.csv']
        ps = [os.path.join(d, n) for n in ns]
        df = [pd.read_csv(p) for p in ps]
        df = pd.concat(df).reset_index(drop=True)
        df['treatment'] = df['path'].apply(get_treatment_name)
        df = time_seconds(df)
        # create time
        var_over_time(df, 'time (s)', 'donutness')



    # DENS over CYLR - cang (n = 12)
    elif do == 'dens_cylr_cang':
        p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_cang_df_toroidal-coords.parquet'
        df = pd.read_parquet(p)
        plot_var_over_cylr(df, 'nb_density_15')
        plot_peak_cylr_min1_box(df, 'nb_density_15')
        plot_var_over_cylr_ind(df, 'nb_density_15')
    

    elif do == 'move_stats':
        df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        boxplots_for_min_1(df, 'dv')
        boxplots_for_min_1(df, 'tor_theta_diff')
        #biva:  dv
        #MannwhitneyuResult(statistic=0.0, pvalue=1.6197135918578049e-06)
        #par4:  dv
        #MannwhitneyuResult(statistic=0.0, pvalue=6.249814797861823e-07)
        #cang:  dv
        #MannwhitneyuResult(statistic=0.0, pvalue=1.2587665380600578e-05)
        #biva:  tor_theta_diff
        #MannwhitneyuResult(statistic=255.0, pvalue=1.6197135918578049e-06)
        #par4:  tor_theta_diff
        #MannwhitneyuResult(statistic=300.0, pvalue=6.249814797861823e-07)
        #cang:  tor_theta_diff
        #MannwhitneyuResult(statistic=166.0, pvalue=0.00022958171420233683)