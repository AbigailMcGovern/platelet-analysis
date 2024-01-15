import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plateletanalysis.variables.basic import time_seconds, cyl_bin, time_bin_1_2_5_10, time_minutes, add_terminating
from plateletanalysis.variables.transform import cylindrical_coordinates
from plateletanalysis.variables.measure import add_finite_diff_derivative
from plateletanalysis.analysis.peaks_analysis import smooth_vars, var_over_cylr, groupby_summary_data_mean
from collections import defaultdict
from toolz import curry
import napari
from scipy.signal import find_peaks


# ---------
# Functions
# ---------

# these have been moved to variables.transform module
def toroidal_coordinates(df):
    # get information about max density peak distance from centre
    gb = ['path', 'cylr_bin', 'time_bin']
    data = var_over_cylr(df, 'nb_density_15', gb)
    data_max = path_dist_at_max(data, ['path', 'time_bin'], 'nb_density_15', dist='cylr_bin')
    # get the tor rho coordinate
    #df = tor_rho_coord(data_max, df) # tor has three coords: rho, z, and theta
    df['tor_rho'] = df['cyl_radial'] - 37.5
    # get tor theta coordinate: will go between +90 (floor outer edge) and -90 (floor inner edge)
    df = tor_theta_coord(df)
    # the coordinates have been visually checked... looks very good!! 
    return df


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


def tor_rho_coord(data_max, df):
    for k, grp in df.groupby(['path', 'time_bin']):
        pdata = data_max[(data_max['path'] == k[0]) & (data_max['time_bin'] == k[1])]
        dist_max = pdata['cylr_bin'].values
        idxs = grp.index.values
        vals = grp['cyl_radial'] - dist_max
        df.loc[idxs, 'tor_rho'] = vals
    return df


def tor_theta_coord(df):
    # in degrees 
    # more negative = more central/epithelial
    # more positive = more distal/epithelial
    tan_theta = df['zs'] / df['tor_rho']
    coef = df['tor_rho'] / np.abs(df['tor_rho'])
    df['tor_theta'] = coef * 90 - np.arctan(tan_theta) / np.pi * 180
    return df


def plot_centred_density_peaks(df):
    # return 
    pass


def view_tracks(v, df):
    for k, grp in df.groupby('path'):
        tracks = grp[['particle', 'frame', 'zs', 'ys', 'x_s']].values
        v.add_tracks(tracks, name=k, properties=grp, visible=False)


def var_over_time(df, t, var):
    plt.rcParams['svg.fonttype'] = 'none'
    #df = df[df['time (s)'] < 600]
    data = groupby_summary_data_mean(df, ['path', 'time (s)'], [var, ])
    data = smooth_vars(data, [var, ], gb='path')
    sns.lineplot(data=data, x=t, y=var, hue='path')
    sns.despine()
    plt.show()


def toroidal_movement_density_count_data(df, sp):
    # max peak distance for different time bins
    gb = ['path', 'cylr_bin', 'minute']
    gb1 = ['path', 'minute']
    dens_cylr = var_over_cylr(df, 'nb_density_15', gb, gb1)
    dens_cylr = dens_cylr.sort_values('cylr_bin')
    dens_max = path_dist_at_max(dens_cylr, ['path', 'minute'], 'nb_density_15', dist='cylr_bin')
    # toroidal movement
    data = groupby_summary_data_mean(df, ['path', 'minute'], ['tor_theta_diff', ])
    #data = smooth_vars(data, ['path', ], gb='path')
    # pathwise peaks data
    out = defaultdict(list)
    d_height, dist, width = 0.0005, 300, 1
    for k, grp in data.groupby('path'):
        out['path'].append(k)
        dens_max_path = dens_max[dens_max['path'] == k]
        # max dens and dens dist values for each min
        for t, dgrp in dens_max_path.groupby('minute'):
            max_dist = dgrp['cylr_bin'].values[0]
            max_dens = dgrp['nb_density_15'].values[0]
            out[f'max_dens_{t}'].append(max_dens) 
            out[f'max_dist_{t}'].append(max_dist)
        # density width @ 1 min
        dens = dens_cylr[(dens_cylr['path'] == k) & (dens_cylr['minute'] == 1)]['nb_density_15'].values
        dens_peaks, dens_props = find_peaks(dens, distance=dist, width=width)
        d_w = dens_props['widths'][0]
        out['dens_width_1_min'].append(d_w)
        # measures of toroidal movement

        for t, tgrp in grp.groupby('minute'):
            out[f'tor_theta_diff_{t}'].append(tgrp['tor_theta_diff'].values[0])
        # measures of count and growth
        sml_df = df[df['path'] == k]
        for t, grp in sml_df.groupby('minute'):
            count = len(pd.unique(grp['particle']))
            out[f'count_{t}'].append(count)
            recruitment = len(sml_df[sml_df['tracknr'] == 1])
            out[f'recruitment_{t}'].append(recruitment)
            shedding = len(sml_df[sml_df['terminating'] == True])
            out[f'shedding_{t}'].append(shedding)
            dens = np.nanmean(grp['nb_density_15'])
            out[f'density_{t}'].append(dens)
            stab = np.nanmean(grp['stab'])
            out[f'instability_{t}'].append(dens)
        
    out = pd.DataFrame(out)
    # save
    out.to_csv(sp)
    return out


def plot_tor_vs_other_measures_min(data):
    plt.rcParams['svg.fonttype'] = 'none'
    min_data = convert_to_min(data)
    min_1_data = min_data[min_data['minute'] == 1]
    fig, axs = plt.subplots(1, 4, sharey=True)
    y = 'tor_theta_diff'
    xs = ['max_dens', 'max_dist', 'count', 'recruitment']
    for i, ax in enumerate(axs):
        print(xs[i])
        xval = min_data[xs[i]].values
        lr = stats.linregress(xval, min_data[y].values)
        print(lr)
        lr1 = stats.linregress(min_1_data[xs[i]].values, min_1_data[y].values)
        print(lr1)
        sns.scatterplot(data=min_1_data, y=y, x=xs[i], ax=ax, palette='rocket',) # alpha=0.6,  hue='minute', )
        xval = min_1_data[xs[i]].values
        x0 = xval.min()
        inc = xval.std()
        m = lr1[0]
        b = lr1[1]
        ax.axline((x0, m * x0 + b), (x0 + inc, m * (x0 + inc) + b), ls='--', color='grey')
        sns.despine(ax=ax)
    plt.show()


def convert_to_min(data):
    cols = ['max_dens', 'max_dist', 'tor_theta_diff', 'count', 'recruitment', 'density', 'instability']
    out = defaultdict(list)
    for i in range(10):
        out['path'] = out['path'] + list(data['path'].values)
        out['minute'] = out['minute'] + [i + 1, ] * len(data['path'].values)
        for c in cols:
            cn = c + f'_{i + 1}'
            out[c] = out[c] + list(data[cn].values)
    out = pd.DataFrame(out)
    return out


def plot_density_closure(data):
    ax = sns.lineplot(data=data, x='minute', y='max_dist', hue='path', err_style="bars", palette='rocket', markers=True)
    sns.stripplot(data=data, x='minute', y='max_dist', palette='rocket', edgecolor='white', linewidth=0.3, jitter=True, size=5, )
    sns.despine()
    plt.show()


def plot_tor_vs_other_measures(data):
    plt.rcParams['svg.fonttype'] = 'none'
    #min_data = convert_to_min(data)
    fig, axs = plt.subplots(1, 4, sharey=True)
    y = 'tor_theta_diff_1'
    data['peak_diff'] = data['max_dist_1'] - (data['max_dist_10'] + data['max_dist_9'] + data['max_dist_8'] \
                                              + data['max_dist_7'] + data['max_dist_6']) / 5
    data['average_count'] = (data['count_1'] + data['count_2'] + data['count_3'] + data['count_4'] + \
          data['count_5'] + data['count_6'] + data['count_7'] + data['count_8'] + data['count_9'] + \
            data['count_10'] ) / 10
    xs = ['donutness magnetude', 'latency donutness', 'n embolysis events', 'average size embolysis events (%)']
    for i, ax in enumerate(axs):
        print(xs[i])
        nanfree = data.dropna(subset=[xs[i], ])
        xval = nanfree[xs[i]].values

        lr = stats.linregress(xval, nanfree[y].values)
        print(lr)
        #lr1 = stats.linregress(min_1_data[xs[i]].values, min_1_data[y].values)
        #print(lr1)
        sns.scatterplot(data=data, y=y, x=xs[i], ax=ax, palette='rocket',) # alpha=0.6,  hue='minute', )
        xval = nanfree[xs[i]].values
        x0 = xval.min()
        inc = xval.std()
        m = lr[0]
        b = lr[1]
        ax.axline((x0, m * x0 + b), (x0 + inc, m * (x0 + inc) + b), ls='--', color='grey')
        sns.despine(ax=ax)
    plt.show()


# -------
# Execute
# -------
if __name__ == '__main__':
    #do = 'examine_tor'
    do = 'make_tor_df'
    #do = 'tor_peaks_analysis'
    #do = 'make_extra_fig_1_plot'
    #do = 'tor_analsis_again'
    if do == 'make_tor_df':
        from pathlib import Path
        import os
        # SALINE
        #p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_spherical-coords.parquet'
        # CONTROL
        #p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_ctrl_df.parquet'
        # CANGRELOR
        #p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_cang_df.parquet'
        # BIVALIRUDIN
        #p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_biva_df.parquet'
        # PAR4
        p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_par4--_df.parquet'
        df = pd.read_parquet(p)
        df = df[df['nrtracks'] > 1]
        if 'time (s)' not in df.columns.values:
            df = time_seconds(df)
        if 'cyl_radial' not in df.columns.values:
            df = cylindrical_coordinates(df)
        df['cylr_bin'] = df['cyl_radial'].apply(cyl_bin)
        df['time_bin'] = df['time (s)'].apply(time_bin_1_2_5_10)
        toroidal_coordinates(df)
        df = add_finite_diff_derivative(df, 'tor_theta', 'path')
        pp = Path(p)
        sp = os.path.join(pp.parents[0], pp.stem + '_toroidal-coords.parquet')
        df.to_parquet(sp)
        

    elif do == 'examine_tor':
        sp = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_toroidal-coords-1.parquet'
        df = pd.read_parquet(sp)
        df['tor_theta_diff'] = - df['tor_theta_diff'] * 0.32
        var_over_time(df, 'time (s)', 'tor_theta_diff')
    

    elif do == 'tor_peaks_analysis':
        sp = ['/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_toroidal-coords-1.parquet', 
              '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_ctrl_df_toroidal-coords.parquet']
        df = [pd.read_parquet(p) for p in sp]
        df = pd.concat(df).reset_index(drop=True)
        df = time_minutes(df)
        df = df[df['nrtracks'] > 1]
        df = add_terminating(df)
        df['tor_theta_diff'] = - df['tor_theta_diff'] * 0.32
        sp_data = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_7/tor_movement_density_count_saline_ctrl.csv'
        data = toroidal_movement_density_count_data(df, sp_data)
        plot_tor_vs_other_measures_min(data)

    elif do == 'make_extra_fig_1_plot':
        sp_data = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_7/tor_movement_density_count_saline.csv'
        data = pd.read_csv(sp_data)
        data = convert_to_min(data)
        plot_density_closure(data)


    elif do == 'tor_analsis_again':
        import os
        sn = ['saline_summary_data_gt1tk.csv', 'control_summary_data.csv']
        d = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
        sp = [os.path.join(d, n) for n in sn]
        sum_data = [pd.read_csv(p) for p in sp]
        sum_data = pd.concat(sum_data).reset_index(drop=True)
        sp_data = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_7/tor_movement_density_count_saline_ctrl.csv'
        data = pd.read_csv(sp_data)
        sum_data = sum_data.set_index('path')
        data = data.set_index('path')
        data = pd.concat([data, sum_data], axis=1)
        plot_tor_vs_other_measures(data)


# MIN 1
# max_dens
# LinregressResult(slope=-161.2644566354689, intercept=0.44482707104232344, rvalue=-0.5098276650489928, pvalue=1.4928122319139026e-24, stderr=14.586924162319276, intercept_stderr=0.03653362165290391)
# LinregressResult(slope=-82.59929091595122, intercept=0.5476379944074842, rvalue=-0.22772604711199412, pvalue=0.18828071543791516, stderr=61.48130397451715, intercept_stderr=0.11034030419694629)
# max_dist
# LinregressResult(slope=0.006660984204419185, intercept=-0.06708497712908146, rvalue=0.3637607254734776, pvalue=2.170898861227395e-12, stderr=0.0009143496462047129, intercept_stderr=0.01721584659771998)
# LinregressResult(slope=0.005852703650638717, intercept=0.2546624588880996, rvalue=0.29405513618316614, pvalue=0.08641599358934834, stderr=0.0033115596077852947, intercept_stderr=0.0841535385004304)
# count
# LinregressResult(slope=-1.1665506102827703e-05, intercept=0.06502297803686977, rvalue=-0.07667286633759476, pvalue=0.1523172011479855, stderr=8.131898464105243e-06, intercept_stderr=0.01442496855580328)
# LinregressResult(slope=-6.89467806378487e-05, intercept=0.49371285977701074, rvalue=-0.5918385434151889, pvalue=0.0001805425768770115, stderr=1.6346293062299457e-05, intercept_stderr=0.02553347815470083)
# recruitment
# LinregressResult(slope=-1.867301771677949e-06, intercept=0.0624903680483901, rvalue=-0.07678316870639244, pvalue=0.15172649208286837, stderr=1.2997949418290202e-06, intercept_stderr=0.012939761703394373)
# LinregressResult(slope=-7.618910394341012e-06, intercept=0.4625672012186616, rvalue=-0.4784034327710149, pvalue=0.0036494351170209, stderr=2.43447574566632e-06, intercept_stderr=0.02423577366541201)


# average_count
#LinregressResult(slope=-4.171662740902025e-05, intercept=0.46400436286087055, rvalue=-0.35306948991055404, pvalue=0.03748317818307382, stderr=1.924335970220557e-05, intercept_stderr=0.03276316332889117)

# DONUT
#donutness magnetude
#LinregressResult(slope=0.036446122539962926, intercept=0.31940548050621476, rvalue=0.4584564036999216, pvalue=0.005609361256035725, stderr=0.012298715518897307, intercept_stderr=0.030984297467157283)
#latency donutness
#LinregressResult(slope=0.0004068514986232184, intercept=0.37282417830227577, rvalue=0.36797175958637035, pvalue=0.029642067731643296, stderr=0.00017896630372633054, intercept_stderr=0.01938110324114288)
#n embolysis events
#LinregressResult(slope=-0.00565309513132835, intercept=0.43190599849475036, rvalue=-0.47590034726384767, pvalue=0.0038570833621228494, stderr=0.001818648445416975, intercept_stderr=0.017296976237765337)
#average size embolysis events (%)
#LinregressResult(slope=-0.0050067960077769995, intercept=0.4274003643793313, rvalue=-0.11388561995633886, pvalue=0.6527415358203303, stderr=0.010919335411903933, intercept_stderr=0.15414182665386664)