import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plateletanalysis.variables.basic import time_seconds, cyl_bin, time_bin_1_2_5_10, time_minutes,\
      add_terminating, time_bin_30_3060_1_2_5_10, get_treatment_name, add_tracknr
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
    d_dropped = data_1_min.dropna(subset=(var,))
    order = ['saline', 'bivalirudin', 'PAR4--', 'cangrelor']
    saline = d_dropped[d_dropped['treatment'] == 'saline'][var].values
    biva = d_dropped[d_dropped['treatment'] == 'bivalirudin'][var].values
    par4 = d_dropped[d_dropped['treatment'] == 'PAR4--'][var].values
    cang = d_dropped[d_dropped['treatment'] == 'cangrelor'][var].values
    print('saline mean: ', saline.mean(), ' +/-', saline.std()/ len(saline) ** 0.5)
    print('biva mean: ', biva.mean(), ' +/-', biva.std()/ len(biva) ** 0.5)
    print('par4 mean: ', par4.mean(), ' +/-', par4.std()/ len(par4) ** 0.5)
    print('cang mean: ', cang.mean(), ' +/-', cang.std()/ len(cang) ** 0.5)
    print('Mann Whitney U tests')
    print('biva: ', var)
    print(stats.mannwhitneyu(saline, biva))
    print('par4: ', var)
    print(stats.mannwhitneyu(saline, par4))
    print('cang: ', var)
    print(stats.mannwhitneyu(saline, cang))
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=data_1_min, x='treatment', order=order, y=var, palette='rocket', ax=ax)
    sns.stripplot(data=data_1_min, x='treatment', order=order, y=var, palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    #fig.subplots_adjust(right=0.95, left=0.5, bottom=0.2, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(3, 4)
    plt.show()


# DONUTNESS - cang vs saline
# --------------------------

def var_over_time(df, t, var):
    plt.rcParams['svg.fonttype'] = 'none'
    #df = df[df['time (s)'] < 600]
    df = df.dropna(subset=[var, ])
    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [var, ])
    data = smooth_vars(data, [var, ], gb='path')
    data['treatment'] = data['path'].apply(get_treatment_name)
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(data=data, x=t, y=var, hue='treatment', palette='rocket', ax=ax)
    sns.despine(ax=ax)
    fig.subplots_adjust(right=0.95, left=0.2, bottom=0.175, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(4, 3)
    plt.show()


def donut_boxplot(data, tx, veh):
    plt.rcParams['svg.fonttype'] = 'none'
    tx_val = data[data['treatment'] == tx]['donutness magnetude'].values
    print('mean tx: ', tx_val.mean())
    print('SEM tx: ', tx_val.std() / len(tx_val) ** 0.5)
    veh_val = data[data['treatment'] == veh]['donutness magnetude'].values
    print('mean veh: ', veh_val.mean())
    print('SEM veh: ', veh_val.std() / len(veh_val) ** 0.5)
    print(stats.mannwhitneyu(tx_val, veh_val))
    fig, ax = plt.subplots(1, 1)
    sns.boxplot(data=data, x='treatment', y='donutness magnetude', palette='rocket', ax=ax)
    sns.stripplot(data=data, x='treatment', y='donutness magnetude', palette='rocket', 
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    fig.subplots_adjust(right=0.95, left=0.55, bottom=0.2, top=0.95, wspace=0.45, hspace=0.4)
    fig.set_size_inches(3, 4)
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
    

def count_over_cylr_plots(
        df, 
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-60 s', '60-120 s', '120-300 s', '300-600 s'], 
        ):
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1) #len(bin_order_0)) 
    i = 0
    data = df.groupby(['path', 'cylr_bin', 'time_bin']).apply(count).reset_index().rename(columns={0 : 'count'})
    data = smooth_vars(data, ['count', ], w=4, gb=['path', 'time_bin'], t='cylr_bin')
    data = data.dropna()
    sns.lineplot(data=data, x='cylr_bin', y='count', ax=ax, hue=bin_col_0, hue_order=bin_order_0, palette='rocket')#hue=hue)
    sns.despine(ax=ax)
    ax.set_xlim(0, 75)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def count_dist_rec_box(
        df, 
        #hue,
        bin_col_0='time_bin', 
        bin_order_0=['0-60 s', '60-120 s', '120-300 s', '300-600 s'], 
        ):
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 1) #len(bin_order_0)) 
    i = 0
    data = df.groupby(['path', 'cylr_bin', 'time_bin']).apply(count).reset_index().rename(columns={0 : 'count'})
    data = smooth_vars(data, ['count', ], w=4, gb=['path', 'time_bin'], t='cylr_bin')
    data = data.dropna()
    data = path_dist_at_max(data, ['path', 'time_bin'], 'count', dist='cylr_bin')
    print(data[data['time_bin'] == '0-60 s']['cylr_bin'].mean())
    print(data[data['time_bin'] == '0-60 s']['cylr_bin'].sem())
    sns.boxplot(data=data, x='time_bin', y='cylr_bin', ax=ax, order=bin_order_0, palette='rocket')#hue=hue)
    sns.stripplot(data=data, x='time_bin',  y='cylr_bin', palette='rocket', order=bin_order_0,
                  edgecolor='white', linewidth=0.3, jitter=True, size=5, ax=ax)
    sns.despine(ax=ax)
    fig.set_size_inches(3.5, 3)
    fig.subplots_adjust(right=0.97, left=0.17, bottom=0.15, top=0.9, wspace=0.4, hspace=0.4)
    plt.show()


def count(grp):
    return len(grp)




def inj_and_plt(d, n):
    p = os.path.join(d, n)
    df = pd.read_parquet(p)
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['nrtracks']>1]
    print(n)
    for tx, grp in df.groupby('treatment'):
        print(tx)
        print(len(pd.unique(grp.path)))
        print(len(pd.unique(grp.particle)))

# -------
# Execute
# -------
if __name__ == '__main__':
    #do = 'tor_movement' # DONE
    #do = 'local_cont'
    #do = 'donutness'
    #do = 'dens_cylr_cang'
    #do = 'move_stats'
    do = 'first_tracked'


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
        df = load_data_all()
        df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        df.to_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        df = local_contraction(df)
        #df['nb_cont_15'] = - df['nb_cont_15'] * 0.32
        #boxplots_for_min_1(df, 'nb_cont_15')
        var_over_time_spec(df, 'nb_cont_15')

        
        # saline mean:  0.0008514012748299563  +/- 0.00021603144455099873
        # biva mean:  -0.0016288318704958799  +/- 0.001214011667603469
        # par4 mean:  0.0010546952617710072  +/- 0.001573157509428031
        # cang mean:  0.003981606032673609  +/- 0.0026567696012408867
         
        # Mann Whitney U tests
        # biva:  nb_cont_15
        # MannwhitneyuResult(statistic=178.0, pvalue=0.05900752319680944)
        # par4:  nb_cont_15
        # MannwhitneyuResult(statistic=185.0, pvalue=0.2501438712743007)
        # cang:  nb_cont_15
        # MannwhitneyuResult(statistic=80.0, pvalue=0.6429683678468101)


    # DONUTNESS - cang (n = 12) vs saline (n = 15)
    elif do == 'donutness':
        #do_tx = 'mips'
        #do_tx = 'cang'
        do_tx = 'SQ'
        # CANG
        if do_tx == 'cang':
            tx = 'cangrelor'
            veh = 'saline'
            ns = ['cangrelor_donut_data_scaled_sn200_n100_c50.csv', 
                  'saline_donut_data_scaled_sn200_n100_c50_gt1tks.csv']
        # MIPS - MannwhitneyuResult(statistic=199.0, pvalue=0.6311267549617171)
        # mean tx:  2.321216447879214
        #SEM tx:  0.19514034351655238
        #mean veh:  2.514396269138107
        #SEM veh:  0.28475841995366097
        elif do_tx == 'mips':
            tx = 'MIPS'
            veh = 'DMSO (MIPS)'
            ns = ['DMSO(MIPS)_and_MIPS_2023_donut_data_scaled_sn200_n100_c50.csv', 
                  'DMSO(MIPS)_donut_data_scaled_sn200_n100_c50.csv', 
                  'MIPS_donut_data_scaled_sn200_n100_c50.csv']
        # SQ - MannwhitneyuResult(statistic=49.0, pvalue=0.3098652593617933)
        # mean tx:  1.4644878723779906
        # SEM tx:  0.2860148629260425
        # mean veh:  1.3889950272663223
        # SEM veh:  0.16874537963048192
        # MannwhitneyuResult(statistic=49.0, pvalue=0.3098652593617933)
        elif do_tx == 'SQ':
            tx = 'SQ'
            veh = 'DMSO (SQ)'
            ns = ['SQ_donut_data_scaled_sn200_n100_c50.csv', 
                  'DMSO(SQ)_donut_data_scaled_sn200_n100_c50.csv']
        
        # data for box plots
        p = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data/231012_mega_summary_data.csv'
        sum_data = pd.read_csv(p)
        sum_data['treatment'] = sum_data['path'].apply(get_treatment_name)
        
        # create box
        sum_data = sum_data[(sum_data['treatment'] == veh) | (sum_data['treatment'] == tx)]
        donut_boxplot(sum_data, tx, veh)
        
        # data for time plots
        d = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
        ps = [os.path.join(d, n) for n in ns]
        df = [pd.read_csv(p) for p in ps]
        df = pd.concat(df).reset_index(drop=True)
        df['treatment'] = df['path'].apply(get_treatment_name)
        df = df[df['treatment'] != 'DMSO (salgav)']
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
        boxplots_for_min_1(df, 'dvz')
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

    elif do == 'first_tracked':
        sp = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_toroidal-coords-1.parquet'
        df = pd.read_parquet(sp)
        df = df[df['tracknr'] == 1]
        df['time_bin'] = df['time (s)'].apply(time_bin_1_2_5_10)
        df['cylr_bin'] = df['cyl_radial'].apply(cyl_bin)
        #count_over_cylr_plots(df)
        count_dist_rec_box(df)
        # 42.10884353741497
        # 1.0129301629384164
