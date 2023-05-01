import pandas as pd
import numpy as np
from scipy import stats
#from plateletanalysis.variables.basic import size_var, inside_injury_var
from toolz import curry

# --------
# Figure 1
# --------




# --------
# Figure 2
# --------

# want to establish that MIPS is not significantly different at 0 seconds

def compare_two_points(
        df, 
        out,
        var, 
        save_path,
        t0=0, 
        t1=300, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        bin_half_width=6
        ):
    results = {
        'time from peak count (s)' : [], 
        'treatment' : [], 
        'U': [], 
        'p' : [], 
        'mean_tx' : [], 
        'std_tx' : [], 
        'mean_ctl' : [], 
        'std_ctl' : [], 
        'tx_n' : [], 
        'ctl_n' : []
    }
    time_from_peak_var(treatments, controls, out, df)
    print(df.columns.values)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        tx_0 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] > t0 - bin_half_width) & (df['time from peak count (s)'] < t0 + bin_half_width)].groupby('path').mean()[var].values
        tx_1 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] > t1 - bin_half_width) & (df['time from peak count (s)'] < t1 + bin_half_width)].groupby('path').mean()[var].values
        ctl_0 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] > t0 - bin_half_width) & (df['time from peak count (s)'] < t0 + bin_half_width)].groupby('path').mean()[var].values
        ctl_1 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] > t1 - bin_half_width) & (df['time from peak count (s)'] < t1 + bin_half_width)].groupby('path').mean()[var].values
        # point 0 
        res0 = stats.mannwhitneyu(tx_0, ctl_0)
        results['time from peak count (s)'].append(t0)
        results['treatment'].append(tx)
        results['U'].append(res0.statistic)
        results['p'].append(res0.pvalue)
        results['mean_tx'].append(tx_0.mean())
        results['mean_ctl'].append(ctl_0.mean())
        results['std_tx'].append(tx_0.std())
        results['std_ctl'].append(ctl_0.std())
        results['tx_n'].append(len(tx_0))
        results['ctl_n'].append(len(ctl_0))
        # point 1
        res1 = stats.mannwhitneyu(tx_1, ctl_1)
        results['time from peak count (s)'].append(t1)
        results['treatment'].append(tx)
        results['U'].append(res1.statistic)
        results['p'].append(res1.pvalue)
        results['mean_tx'].append(tx_1.mean())
        results['mean_ctl'].append(ctl_1.mean())
        results['std_tx'].append(tx_1.std())
        results['std_ctl'].append(ctl_1.std())
        results['tx_n'].append(len(tx_1))
        results['ctl_n'].append(len(ctl_1))
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results


def time_from_peak_var(treatments, controls, out, summary_data):
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        t = out[out['treatment'] == ctl]['time peak count'].mean()
        for k, g in summary_data.groupby('treatment', group_keys=False):
        #t = out[out['treatment'] == k]['time peak count'].mean()
            if k in [ctl, tx]:
                vs = g['time (s)'].values - t
                idxs = g.index.values
                summary_data.loc[idxs, 'time from peak count (s)'] = vs



def compare_two_phases(
        df, 
        out,
        var, 
        save_path,
        t=0,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        bin_half_width=6
        ):
    results = instantiate_results()
    time_from_peak_var(treatments, controls, out, df)
    print(df.columns.values)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        tx_0 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] < t)].groupby('path').mean()[var].values
        tx_1 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] > t)].groupby('path').mean()[var].values
        ctl_0 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] < t)].groupby('path').mean()[var].values
        ctl_1 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] > t)].groupby('path').mean()[var].values
        # point 0 
        add_mann_whitney_u_data(tx, t, tx_0, ctl_0, results)
        # point 1
        add_mann_whitney_u_data(tx, t, tx_1, ctl_1, results)
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results


def outlier_test(
        df, 
        save_path,
        ):
    res = {
        'path' : [], 
        'treatment' : [],
        'platelet count' : [], 
    }
    for k, grp in df.groupby(['path', 'treatment']):
        res['path'].append(k[0])
        res['treatment'].append(k[1])
        res['platelet count'].append(grp['platelet count'].mean())
    res = pd.DataFrame(res)
    outliers = {
        'path' : [], 
        'treatment' : [], 
        'Q1' : [], 
        'Q3' : [],
        'platelet count' : []
    }
    for k, grp in res.groupby('treatment'):
        v = grp['platelet count'].values
        paths = grp['path'].values
        Q1 = stats.scoreatpercentile(v, 25)
        Q3 = stats.scoreatpercentile(v, 75)
        IQR = Q3 - Q1
        ll = Q1 - 1.5 * IQR
        ul = Q3 + 1.5 * IQR
        l_idx = np.where(v < ll)
        u_idx = np.where(v > ul)
        idxs = [l_idx, u_idx]
        include = []
        for i, idl in enumerate(idxs):
            if np.sum(idl) > 0:
                include.append(1)
            else:
                include.append(0)
        if np.sum(include) == 2:
            idx = np.concatenate([l_idx[0], u_idx[0]])
            idx = (idx, )
        elif np.sum(include) == 1:
            for i, idl in enumerate(idxs):
                if np.sum(idl) > 0:
                    idx = idl
        else:
            idx = 0
        if np.sum(idx) > 0:
            ol_count = v[idx]
            ol_path = paths[idx]
            outliers['path'] = np.concatenate([outliers['path'], ol_path])
            outliers['platelet count'] = np.concatenate([outliers['platelet count'], ol_count])
            Q1s = [Q1, ] * len(ol_count)
            outliers['Q1'] = np.concatenate([outliers['Q1'], Q1s])
            Q3s = [Q3, ] * len(ol_count)
            outliers['Q3'] = np.concatenate([outliers['Q3'], Q3s])
            txs = [k, ] * len(ol_count)
            outliers['treatment'] = np.concatenate([outliers['treatment'], txs])
    outliers = pd.DataFrame(outliers)
    outliers.to_csv(save_path)




# --------
# Figure 3
# --------

# 

def compare_two_phases_and_sizes(
        df, 
        out,
        var, 
        save_path,
        t=0,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        ):
    results = instantiate_results(other=('thrombus size', ))
    time_from_peak_var(treatments, controls, out, df)
    print(df.columns.values)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        for s in ('large', 'small'):
            tx_0 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] < t) & (df['thrombus size'] == s)].groupby('path').mean()[var].values
            tx_1 = df[(df['treatment'] == tx) & (df['time from peak count (s)'] > t) & (df['thrombus size'] == s)].groupby('path').mean()[var].values
            ctl_0 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] < t) & (df['thrombus size'] == s)].groupby('path').mean()[var].values
            ctl_1 = df[(df['treatment'] == ctl) & (df['time from peak count (s)'] > t) & (df['thrombus size'] == s)].groupby('path').mean()[var].values
            # point 0 
            add_mann_whitney_u_data(tx, t, tx_0, ctl_0, results, gt=False, other={'thrombus size' : s})
            # point 1
            add_mann_whitney_u_data(tx, t, tx_1, ctl_1, results, gt=True, other={'thrombus size' : s})
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results


def compare_two_phases_and_insideout(
        df, 
        var, 
        save_path,
        t=0,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        ):
    results = instantiate_results(other=('location', ))
    print(df.columns.values)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        for s in (True, False):
            if s:
                n = 'inside injury'
            else:
                n = 'outside injury'
            tx_0 = df[(df['treatment'] == tx) & (df['time from peak count'] < t) & (df['inside injury'] == s)].groupby('path').mean()[var].values
            tx_1 = df[(df['treatment'] == tx) & (df['time from peak count'] > t) & (df['inside injury'] == s)].groupby('path').mean()[var].values
            ctl_0 = df[(df['treatment'] == ctl) & (df['time from peak count'] < t) & (df['inside injury'] == s)].groupby('path').mean()[var].values
            ctl_1 = df[(df['treatment'] == ctl) & (df['time from peak count'] > t) & (df['inside injury'] == s)].groupby('path').mean()[var].values
            # point 0 
            add_mann_whitney_u_data(tx, t, tx_0, ctl_0, results, gt=False, other={'location' : n})
            # point 1
            add_mann_whitney_u_data(tx, t, tx_1, ctl_1, results, gt=True, other={'location' : n})
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results


def compare_two_phases_size_and_insideout(
        df, 
        var, 
        save_path,
        t=0,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        ):
    results = instantiate_results(other=('location', 'size'))
    print(df.columns.values)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        for s in (True, False):
            if s:
                n = 'inside injury'
            else:
                n = 'outside injury'
            tx_0 = df[(df['treatment'] == tx) & (df['time from peak count'] < t) & (df['inside injury'] == s)]
            tx_1 = df[(df['treatment'] == tx) & (df['time from peak count'] > t) & (df['inside injury'] == s)]
            ctl_0 = df[(df['treatment'] == ctl) & (df['time from peak count'] < t) & (df['inside injury'] == s)]
            ctl_1 = df[(df['treatment'] == ctl) & (df['time from peak count'] > t) & (df['inside injury'] == s)]
            for sz in ['large', 'small']:
                tx_0_sz = tx_0[tx_0['size'] == sz].groupby('path').mean()[var].values
                tx_1_sz = tx_1[tx_1['size'] == sz].groupby('path').mean()[var].values
                ctl_0_sz = ctl_0[ctl_0['size'] == sz].groupby('path').mean()[var].values
                ctl_1_sz = ctl_1[ctl_1['size'] == sz].groupby('path').mean()[var].values
                # point 0 
                add_mann_whitney_u_data(tx, t, tx_0_sz, ctl_0_sz, results, gt=False, other={'location' : n, 'size' : sz})
                # point 1
                add_mann_whitney_u_data(tx, t, tx_1_sz, ctl_1_sz, results, gt=True, other={'location' : n, 'size' : sz})
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results

# -----
# CMAPS
# -----
import seaborn as sns
import matplotlib.pyplot as plt
MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  = dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1 = {**pal_MIPS,**pal_cang,**pal_SQ}


def insideout_plots(
        df,
        treatment='MIPS', 
        control='DMSO (MIPS)',
        ):
    sns.set_context('paper')
    sns.set_style('ticks')
    df = pd.concat([df[df['treatment'] == treatment], df[df['treatment'] == control]])
    df['phase'] = df['time from peak count'].apply(_growth_consol)
    df['location'] = df['inside injury'].apply(_insideout_str)
    df['phase x location'] = df['phase'] + ': ' + df['location']
    res = {
        'path': [], 
        'treatment' : [], 
        'phase x location' : [], 
        'platelet count' : [], 
        'platelet density um^-3' : [], 
    }
    for k, grp in df.groupby(['path', 'treatment', 'phase x location']):
        res['path'].append(k[0])
        res['treatment'].append(k[1])
        res['phase x location'].append(k[2])
        res['platelet count'].append(grp['platelet count'].mean())
        res['platelet density um^-3'].append(grp['platelet density um^-3'].mean())
    res = pd.DataFrame(res)
    fig, axs = plt.subplots(1, 2)
    _ord = ['growth: inside injury', 'growth: outside injury', 'consolidation: inside injury', 'consolidation: outside injury']
    hue_ord = ['DMSO (MIPS)', 'MIPS']
    sns.barplot(data=res, x='phase x location', y='platelet count', hue='treatment', palette=pal1,
                order=_ord, hue_order=hue_ord, ax=axs[0], capsize=.15, linewidth=0.5, errorbar='se')
    for label in axs[0].get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    sns.barplot(data=res, x='phase x location', y='platelet density um^-3', hue='treatment', palette=pal1, 
                order=_ord, hue_order=hue_ord, ax=axs[1], capsize=.15, linewidth=0.5, errorbar='se')
    for label in axs[1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    sns.despine()
    fig.subplots_adjust(right=0.95, left=0.12, bottom=0.5, top=0.95, wspace=0.5, hspace=0.4)
    fig.set_size_inches(7, 3)
    plt.show()


def _insideout_str(val):
    if val:
        return 'inside injury'
    else:
        return 'outside injury'

def _growth_consol(val):
    if val > 0:
        return 'consolidation'
    else:
        return 'growth'

# -------------
# Figures 4 & 5
# -------------


def compare_phases_sizes_regions(
        out,
        rdf,
        summary_data,
        var, 
        save_path,
        t=0,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        region_var='region'
        ):
    results = instantiate_results(other=('thrombus size', 'region'))
    df = unionise_regions_time_size_data(summary_data, rdf, treatments, controls, out)
    print('time from peak count (s): ', df['time from peak count (s)'].min(), df['time from peak count (s)'].max())
    print(df.columns.values)
    print(pd.unique(df[region_var]))
    print(pd.unique(df['treatment']))
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        for s in ('large', 'small'):
            for r in regions:
                sdf = df[(df['treatment'] == tx) & \
                            (df['thrombus size'] == s)]
                l = len(sdf)
                tx_0_gbm = df[(df['treatment'] == tx) & \
                          (df['time from peak count (s)'] < t) & \
                          (df[region_var] == r) & \
                            (df['thrombus size'] == s)].groupby('path').mean(numeric_only=True)
                tx_0 = tx_0_gbm[var].values
                tx_1_gbm = df[(df['treatment'] == tx) & \
                          (df['time from peak count (s)'] > t) & \
                          (df[region_var] == r) & \
                            (df['thrombus size'] == s)].groupby('path').mean(numeric_only=True)
                tx_1 = tx_1_gbm[var].values
                ctl_0_gbm = df[(df['treatment'] == ctl) & \
                           (df['time from peak count (s)'] < t) & \
                          (df[region_var] == r) & \
                            (df['thrombus size'] == s)].groupby('path').mean(numeric_only=True)
                ctl_0 = ctl_0_gbm[var].values
                ctl_1_gbm = df[(df['treatment'] == ctl) & \
                           (df['time from peak count (s)'] > t) & \
                          (df[region_var] == r) & \
                            (df['thrombus size'] == s)].groupby('path').mean(numeric_only=True)
                ctl_1 = ctl_1_gbm[var].values
                # point 0 
                add_mann_whitney_u_data(tx, t, tx_0, ctl_0, results, gt=False, other={'thrombus size' : s, 'region' : r})
                # point 1
                add_mann_whitney_u_data(tx, t, tx_1, ctl_1, results, gt=True, other={'thrombus size' : s,  'region' : r})
                print(f'Completed stats for {tx} vs {ctl} for {s} thrombi in {r} region')
    results = pd.DataFrame(results)
    results.to_csv(save_path)
    return results



def unionise_regions_time_size_data(summary_data, rdf, treatments, controls, out):
    time_from_peak_var(treatments, controls, out, rdf)
    func = get_sizes(summary_data)
    rdf['thrombus size'] = rdf['path'].apply(func)
    return rdf
    # rdf has n path x n regions
    # df has n path

@curry
def get_sizes(summary_data, path):
    return summary_data[summary_data['path'] == path]['thrombus size'].values[0] 




# --------
# Figure 6
# --------

def correlate_tracknr_w_density_large():
    pass


# -------
# Helpers
# -------

def instantiate_results(other=()):
    results = {
        'time from peak count (s)' : [], 
        'treatment' : [], 
        'U': [], 
        'p' : [], 
        'mean_tx' : [], 
        'std_tx' : [], 
        'mean_ctl' : [], 
        'std_ctl' : [], 
        'tx_n' : [], 
        'ctl_n' : [], 
    }
    for k in other:
        results[k] = []
    return results


def add_mann_whitney_u_data(tx, t, tx_data, ctl_data, results, gt=True, other={}):
    res1 = stats.mannwhitneyu(tx_data, ctl_data)
    if gt:
        results['time from peak count (s)'].append(f'> {t}')
    else:
        results['time from peak count (s)'].append(f'< {t}')
    results['treatment'].append(tx)
    results['U'].append(res1.statistic)
    results['p'].append(res1.pvalue)
    results['mean_tx'].append(tx_data.mean())
    results['mean_ctl'].append(ctl_data.mean())
    results['std_tx'].append(tx_data.std())
    results['std_ctl'].append(ctl_data.std())
    results['tx_n'].append(len(tx_data))
    results['ctl_n'].append(len(ctl_data))
    for k in other.keys():
        results[k].append(other[k])
    


if __name__ == '__main__':
    # is the data from today (i.e., filename with today's date)
    data_today = False

    from datetime import datetime
    now = datetime.now()
    date = now.strftime("%y%m%d")
    today = date

    if not data_today:
        date = 230421 # input the date on the file you want to use

    # Which stats to do
    fig2 = False
    fig3 = False
    fig4 = False

    outliers = True

    # use data with inside injury variable ("inside injury" : bool) - for Fig 3
    insideout = False
    if not insideout:
        nstr = ''
    else:
        nstr = '_insideout'

    # DATA
    sum_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_rolling-counts.csv'
    summary_data = pd.read_csv(sum_p)
    out_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data.csv'
    out = pd.read_csv(out_p)
    out_g_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data-growth.csv'
    out_g = pd.read_csv(out_g_p)
    out_c_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/{date}_count-and-growth-pcnt{nstr}_centile-data-consolidation.csv'
    out_c = pd.read_csv(out_c_p)
    rd_p = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230422_regionsdata_9var_trk1_seconds_sizes.csv'
    rdf = pd.read_csv(rd_p)
    #size = rdf['size'].values
    #rdf = rdf.drop(columns='size')
    #rdf['size'] = size

    # Figure 2
    if fig2:
        save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_0-vs400_count.csv'
        save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_0-vs400_density.csv'
        compare_two_points(summary_data, out, 'platelet count', save_path_0)
        compare_two_points(summary_data, out, 'density (platelets/um^2)', save_path_1)
        save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol_count.csv'
        save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol_density.csv'
        compare_two_phases(summary_data, out, 'platelet count', save_path_0)
        compare_two_phases(summary_data, out, 'density (platelets/um^2)', save_path_1)


    # Figure 3
    if fig3:
        save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol-vs-size_count.csv'
        save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol-vs-size_density.csv'
        compare_two_phases_and_sizes(summary_data, out, 'platelet count', save_path_0)
        compare_two_phases_and_sizes(summary_data, out, 'density (platelets/um^2)', save_path_1)
    
    # Figure 3 - insideout
    if insideout:
        sp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230429_inside_outside_size_counts_density.csv'
        df = pd.read_csv(sp)
        #save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{today}_growth-vs-consol-vs-size-vs-insideout_count.csv'
        #save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{today}_growth-vs-consol-vs-size-vs-insideout_density.csv'
        #compare_two_phases_and_insideout(df, 'platelet count', save_path_0)
        #compare_two_phases_and_insideout(df, 'platelet density um^-3', save_path_1)
        #insideout_plots(df)
        save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{today}_growth-vs-consol-vs-SIZE-vs-insideout_count.csv'
        save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{today}_growth-vs-consol-vs-SIZE-vs-insideout_density.csv'
        compare_two_phases_size_and_insideout(df, 'platelet count', save_path_0)
        compare_two_phases_size_and_insideout(df, 'platelet density um^-3', save_path_1)

        

    # Figure 4
    if fig4:
        save_path_0 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol-vs-size-vs-region_count.csv'
        save_path_1 = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_growth-vs-consol-vs-size-vs-region_density.csv'
        compare_phases_sizes_regions(out, rdf, summary_data, 'platelet count', save_path_0)
        compare_phases_sizes_regions(out, rdf, summary_data, 'platelet density (um^-3)', save_path_1)

    if outliers:
        save_path = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/statistics/{date}_outliers.csv'
        outlier_test(summary_data, save_path)
