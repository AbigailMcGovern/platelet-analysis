import pandas as pd
import numpy as np
from scipy import stats
from toolz import curry

# -------------------------------
# Growth and consolidation phases
# -------------------------------

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



# ------------------------------
# Compare size and inside injury
# ------------------------------

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



# ---------------
# Compare regions
# ---------------

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


def phase_region_multivar(
        df, 
        save_path,
        variables,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline') ,
        regions=('center', 'anterior', 'lateral', 'posterior'), 
        region_var='region', 
        phase_var='phase',
        phases=('growth', 'consolidation')
        ):
    other = variables + ('region', )
    results = instantiate_results(other=other)
    for i, tx in enumerate(treatments):
        ctl = controls[i]
        for k, grp in df.groupby(['phase', 'region'] ):
            for v in variables:
                pass




# -------------
# Outlier tests
# -------------

def IQR_outlier_identification(
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




# --------------
# Base functions
# --------------


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
    

