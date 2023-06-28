import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from plateletanalysis.variables.basic import add_region_category
import os
from toolz import curry
from plateletanalysis.variables.measure import quantile_normalise_variables
from scipy.signal import find_peaks #, peak_widths, peak_prominences
import seaborn as sns
from time import time
import matplotlib.pyplot as plt


# MAIN FUNCS
# - rolling_counts_and_growth_plus_peaks (figures 2, 3)
# - quantile_analysis_data (figure 3)
# - inside_and_outside_injury_counts_density (figure 3)
# - inside_and_outside_injury_counts_density_thrombus_size (figure 3)
# - experiment_phase_insideout_data 
# - angle_binned_outside_injury_phase_data (fig 3)
# - experiment_time_region_data (figures 4, 5)
# - experiment_region_phase_data (fig 4)
# - cylrad_binned_region_phase_data (fig 4)


# -----------------
# Summary Variables
# -----------------

# USED IN QUANTILES

def _count_max(grp, col):
    return grp[col].max()

def _growth_max(grp, col):
    return grp[col].max()

def _AUC_count(grp, col):
    return grp[col].sum()

def _count_mean(grp, col):
    return grp[col].mean()

def _time_to_peak(grp, col):
    t = grp[col].values[0]
    return t

def _growth_mean(grp, col):
    return grp[col].mean()


# USED IN REGIONS

@curry
def platelet_mean_of_var(var, grp):
    means = grp.groupby('particle')[var].mean()
    return np.nanmean(means)

@curry
def frame_mean_of_var(var, grp):
    means = grp.groupby('frame')[var].mean()
    return np.nanmean(means)

def phase_count(grp):
    cs = grp.groupby('frame').apply(count)
    return np.mean(cs)


def count(grp):
    return len(pd.unique(grp.particle))


@curry
def density(dens_col, grp, *args):
    return np.mean(grp[dens_col].values)

@curry
def outer_edge(dist_col, pcnt_lims, grp, *args):
    grp = _exclude_back_quadrant(grp)
    grp = quantile_normalise_variables(grp, [dist_col, ]) # does framewise within loop_over_exp
    dist_pcnt_col = dist_col + '_pcnt'
    grp = grp[(grp[dist_pcnt_col] > pcnt_lims[0]) & (grp[dist_pcnt_col] < pcnt_lims[1])]
    return np.mean(grp[dist_col].values)


def _exclude_back_quadrant(df, col='phi', lim=- np.pi / 2):
    df = df[df[col] > lim]
    return df


def recruitment(df):
    one_track = df[df['tracknr'] == 1]
    val = len(pd.unique(one_track.particle)) / 0.321764322705706
    return val


def recruitment_phase(df):
    '''number recruited during phase'''
    one_track = df[df['tracknr'] == 1]
    val = len(pd.unique(one_track.particle))
    return val


def shedding_phase(df):
    '''Number shed during phase'''
    term = df[df['terminating'] == True]
    val = len(pd.unique(term.particle))
    return val


@curry
def where_shed(coord, df):
    '''
    Doesnt work if data is binned by the coord
    (unless binned by where recruited in coord)
    '''
    term = df[df['terminating'] == True]
    val = term[coord].mean()
    return val


@curry
def where_recruited(coord, df):
    '''
    Doesnt work if data is binned by where recruited in the coord
    '''
    one_track = df[df['tracknr'] == 1]
    val = one_track[coord].mean()
    return val


def shedding(df):
    return df['terminating'].sum() / 0.321764322705706


def stability(df):
    return df['stab'].mean()


def dvy(df):
    return df['dvy'].mean()


def tracking_time(df):
    nframes = df['tracknr'].mean()
    return nframes / 0.321764322705706


def platelet_average_tracking_time(df):
    nframes = df.groupby('particle')['tracknr'].mean()
    ttime = nframes / 0.321764322705706
    return np.mean(ttime)


def sliding(df):
    return df['sliding (ums^-1)'].mean()


def total_sliding_phase(df):
    '''Total um sliding'''
    return df['sliding (ums^-1)'].sum() * 0.321764322705706


def average_sliding_phase(df):
    '''average um sliding per platelet'''
    plt_sliding = df.groupby('particle')['sliding (ums^-1)'].sum()
    return np.mean(plt_sliding)


def p_lt15s(df):
    lt15 = len(df[df['total time tracked (s)'] < 15])
    t = len(df)
    #gt30 = len(df[df['time (s)'] >= 30])
    return lt15 / t


def p_lt15_phase(df):
    t = len(pd.unique(df['particle']))
    lt15 = len(pd.unique(df[df['nrtracks'] < 5]['particle']))
    return lt15 / t


def p_gt60_phase(df):
    t = len(pd.unique(df['particle']))
    gt60 = len(pd.unique(df[df['nrtracks'] > 19]['particle']))
    return gt60 / t


def p_gt60s(df):
    gt60 = len(df[df['total time tracked (s)'] > 60])
    t = len(df)
    #gt30 = len(df[df['time (s)'] >= 30])
    return gt60 / t


def tracking_time_IQR(df):
    Q1 = stats.scoreatpercentile(df['tracking time (s)'].values, 25)
    Q3 = stats.scoreatpercentile(df['tracking time (s)'].values, 75)
    return Q3 - Q1


def p_shed_lt15(df):
    p = len(df[(df['total time tracked (s)'] < 15) & (df['terminating'] == True)])
    t = len(df[df['terminating'] == True])
    return p / t


def p_shed_gt60(df):
    p = len(df[(df['total time tracked (s)'] > 60) & (df['terminating'] == True)])
    t = len(df[df['terminating'] == True])
    return p / t


def p_recruited_lt15(df):
    #sdf = df[df['total time tracked (s)'] < 15]
    #sdf = sdf[df['tracknr'] == 1])]
    p = len(df[(df['total time tracked (s)'] < 15) & (df['tracknr'] == 1)])
    t = len(df[df['tracknr'] == 1])
    try:
        v = p / t
    except:
        v = np.NaN
    return v


def p_recruited_gt60(df):
    #sdf = df[df['total time tracked (s)'] < 15]
    #sdf = sdf[df['tracknr'] == 1])]
    p = len(df[(df['total time tracked (s)'] > 60) & (df['tracknr'] == 1)])
    t = len(df[df['tracknr'] == 1])
    try:
        v = p / t
    except:
        v = np.NaN
    return v


def densification(grp):
    grp = grp.sort_values('time (s)')
    fde = grp.groupby('particle')['nb_density_15'].diff()
    grp1 = grp.copy()
    grp1['ddif'] = fde
    sums = grp1.groupby('particle')['ddif'].sum()
    return np.mean(sums)


@curry
def cumulative(var, grp):
    cumsum = grp.groupby('particle')[var].apply(np.nansum)
    return np.nanmean(cumsum)


@curry
def var_for_first_3_frames(var, grp):
    '''
    This one will only work when entire platelet track
    is binned into single bin
    '''
    f3f = grp[grp['tracknr'] < 4]
    vals = f3f.groupby('particle')[var].mean()
    return np.mean(vals)


def initial_platelet_densification(grp):
    f3f = grp[grp['tracknr'] < 4]
    fde = f3f.groupby('particle')['nb_density_15'].diff()
    grp1 = grp.copy()
    grp1['ddif'] = fde
    sums = grp1.groupby('particle')['ddif'].sum()
    return np.nansum(sums)


def initial_platelet_velocity_change(grp):
    f3f = grp[grp['tracknr'] < 4]
    fde = f3f.groupby('particle')['dv'].diff()
    grp1 = grp.copy()
    grp1['ddif'] = fde
    sums = grp1.groupby('particle')['ddif'].sum()
    return np.nansum(sums)



def frame_average_net_platelet_loss(grp):
    grp = grp.sort_values('time (s)')
    count = grp.groupby('time (s)')['particle'].apply(len)
    diff = count.diff()
    val = - diff.mean() * 0.321764322705706 * 60
    return val



def average_greatest_net_platelet_loss(grp):
    grp = grp.sort_values('time (s)')
    count = grp.groupby('time (s)')['particle'].apply(len)
    diff = count.diff()
    diff = diff.rolling(window=8,center=False).mean()
    max_ = diff.min()
    return - max_ * 0.321764322705706 * 60



def number_lost(grp):
    first = grp[grp['tracknr'] == 1]
    grp['term'] = grp['tracknr'] == grp['nrtracks']
    last = grp[grp['term'] == True]
    return len(last) - len(first) 


@curry
def cumulative_var(var, grp):
    sums = grp.groupby('particle')[var].sum()
    return sums.mean()



def n_platelets_from_other(grp):
    particles = np.array([])
    n_new_other = []
    grp = grp[grp['tracknr'] > 1]
    for k, g in grp.groupby('time (s)'):
        ps = pd.unique(g['particle'].values)
        if len(particles) >  0:
            new = np.sum(np.isin(ps, particles, invert=True))
            n_new_other.append(new)
    return np.sum(n_new_other)




# --------------------
# General summary data
# --------------------

def rolling_counts_and_growth_plus_peaks(
        df: pd.DataFrame, 
        save_path, 
        insideout=False, 
        mean_vars=(),
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        time_col='time (s)',
        cat_cols=('treatment', ),
        vars_for_veh_pcnt = ('platelet count raw', 'platelet density (um^-3) raw', 
                         'platelet count', 'platelet density (um^-3)')
        ):
    '''
    Generate experiment by time data frame with rolling counts and rolling growth. 
    Uses time (s) variable. Can add additional variables by adding mean_vars.
    Will also generate a data frame showing peaks in counts troughs in growth.
    Use basic.add_basic_variables_to_files to ensure all necessary vars are in df.
    The function will also save growth and counts plots for individual experiments
    with the annotated peaks (these correspond to the peaks data).

    Parameters
    ----------
    df: pd.DataFrame
        Platelet observation data frame of the type manipulated by variables module
    save_path: str
        Path to which to save output. The path will be used to generate two files, 
        one ending in '_peaks.csv' and the other ending in '_rolling-counts.csv'. 
    insideout: bool
        Option to add extra dimension to the data with 'inside injury' variable. 
        Output will be experiment x time x inside injury (NB: inside injur is bool)
    mean_vars: list like of str
        List of columm names for which to add additional mean data
    
    Returns
    -------
    summary_data: pd.DataFrame
        Experiment x time data frame with rolling counts and rolling growth
    peaks_data: pd.DataFrame
        Experiment df with peaks 
    '''
    peaks_data = {
        'path' : [], 
        'treatment' : [],
        'peak count' : [], 
        'peak growth' : [], 
        'time peak count' : [],
        'time trough growth' : []
    }
    for v in mean_vars:
        peaks_data[v] = []
    summary_data = {
        'path' : [], 
        'treatment' : [], 
        'time (s)' : [],
        'platelet count raw' : [], 
        #'growth (platelets/s)' : [], 
        'size (um) raw' : [], 
        'density (platelets/um^2) raw' : [], 
        'thrombus size' : [], 
    }
    if insideout:
        summary_data['inside injury'] = []
    for v in mean_vars:
        summary_data[v] = []
    print('Obtaining counts...')
    if insideout:
        gb = ['path', 'time (s)', 'treatment', 'inside_injury']
    else:
        gb = ['path', 'time (s)', 'treatment']
    for k, grp in df.groupby(gb):
        summary_data['path'].append(k[0])
        summary_data['time (s)'].append(k[1])
        summary_data['treatment'].append(k[2])
        summary_data['platelet count raw'].append(len(grp))
        sz_df = grp[grp['outer_edge'] == True] # cyl_r 90-98th centile plts
        edge = sz_df['cyl_r'].mean()
        dens = grp['nb_density_15'].mean()
        summary_data['size (um) raw'].append(edge)
        summary_data['density (platelets/um^2) raw'].append(dens)
        summary_data['thrombus size'].append(grp['size'].values[0])
        if insideout:
            summary_data['inside injury'].append(grp['inside_injury'].values[0])
        for v in mean_vars:
            summary_data[v].append(grp[v].mean())
        #summary_data['growth (platelets/s)'].append(grp['rolling_growth'].values[0])
    summary_data = pd.DataFrame(summary_data)
    summary_data = summary_data.sort_values('time (s)') # sorted according to time ... important to roll
    print('Getting rolling counts and growth...')
    for k, grp in summary_data.groupby('path'):
        #grp = grp.sort_values('time (s)')
        _add_rolled_and_diff(grp, 'platelet count raw', 'growth (platelets/s)', summary_data)
        _add_rolled_and_diff(grp, 'size (um) raw', 'growth (um/s)', summary_data)
        _add_rolled_and_diff(grp, 'density (um^-3) raw', 'contraction (platelets/um s^-1)', summary_data)
    print('Finding peaks...')
    for k, grp in summary_data.groupby('path'):
        count_peak_idxs = find_peaks(grp['platelet count'].values)
        growth_peak_idxs = find_peaks(grp['growth (platelets/s)'].values)
        neg_growth_peak_idxs = find_peaks(- grp['growth (platelets/s)'].values)
        if len(count_peak_idxs[0]) > 0:
            count_t = grp['time (s)'].values[count_peak_idxs[0][0]]
            count_v = grp['platelet count'].values[count_peak_idxs[0][0]]
        else:
            count_t = np.NaN
            count_v = np.NaN
        if len(growth_peak_idxs[0]) > 0:
            growth_t = grp['time (s)'].values[neg_growth_peak_idxs[0][0]]
            growth_v = grp['growth (platelets/s)'].values[growth_peak_idxs[0][0]]
        else:
            growth_t = np.NaN
            growth_v = np.NaN
        peaks_data['path'].append(k)
        peaks_data['peak count'].append(count_v)
        peaks_data['peak growth'].append(growth_v)
        peaks_data['time peak count'].append(count_t)
        peaks_data['time trough growth'].append(growth_t)
        peaks_data['treatment'].append(grp['treatment'].values[0])
        for v in mean_vars:
            peaks_data[v].append(grp[v].mean())
        idxs = grp.index.values
        summary_data.loc[idxs, 'time peak count'] = count_t
        summary_data.loc[idxs, 'time  trough growth'] = growth_t
    peaks_data = pd.DataFrame(peaks_data)
    save_peaks = os.path.join(Path(save_path).parents[0], Path(save_path).stem + '_peaks.csv')
    peaks_data.to_csv(save_peaks)
    save_growth_plots(summary_data, peaks_data, save_path)
    save_summary = os.path.join(Path(save_path).parents[0], Path(save_path).stem + '_rolling-counts.csv')
    summary_data.to_csv(save_summary)
    percent_vehicle_data(vars_for_veh_pcnt, summary_data, treatments, controls, cat_cols, time_col)
    return summary_data, peaks_data


def _add_rolled_and_diff(grp, col, diff_n, summary_data):
    idxs = grp.index.values
    roll = grp[col].rolling(window=20,center=False).mean()
    coln = col[:-4]
    summary_data.loc[idxs, coln] = roll
    #grp[diff_n] = roll.diff() * 0.321764322705706
    summary_data.loc[idxs, diff_n] = roll.diff().rolling(window=20,center=False).mean() * 0.321764322705706


def save_growth_plots(summary_data, peaks_data, save_path):
    #treatments = pd.unique(summary_data['treatment'])
    fig, ax = plt.subplots(1, 1)
    for tx, grp in summary_data.groupby('treatment'):
        sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{tx}-count-plot.pdf')
        pk = peaks_data[peaks_data['treatment'] == tx]
        sns.lineplot(data=grp, x='time (s)', y='platelet count', hue='path', ax=ax)
        x, y = pk['time peak count'].values, pk['peak count'].values
        ax.scatter(x, y, s=7)
        fig.savefig(sp)
        ax.clear()
        sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{tx}-growth-plot.pdf')
        sns.lineplot(data=grp, x='time (s)', y='growth (platelets/s)', ax=ax, hue='path')
        #ax.scatter(x=pk['time peak growth'].values, y=pk['peak growth'].values, s=7)
        fig.savefig(sp)
        ax.clear()


def percent_vehicle_data(names, data, treatments, controls, other_cols, time_col):
    for n in names:
        nn = f'{n} pcnt'
        data[nn] = [None, ] * len(data)
    for i, tx in enumerate(treatments):
        v = controls[i]
        tx_df = data[data[other_cols[0]] == tx]
        for t, g in tx_df.groupby([time_col, ]):
            idxs = g.index.values
            v_df = data[(data[other_cols[0]] == v) & (data[time_col] == t)]
            v_means = {n : v_df[n].mean() for n in names}
            for n in names:
                nn = f'{n} pcnt'
                pcnt = g[n].values / v_means[n] * 100
                data.loc[idxs, nn] = pcnt


# ----------------------
# Qunatile analysis data
# ----------------------


def quantile_analysis_data(
        summary_data, 
        peaks_data,
        save_path,
        insideout=False
        ):
    df = df[df['nrtracks'] > 1]
    groupby = ['path', 'treatment',]
    if insideout:
        groupby.append('inside injury')
    funcs = [_count_max, _growth_max, _AUC_count, 
             _count_mean, _time_to_peak, _growth_mean,

             _count_max, _growth_max, _AUC_count, 
             _count_mean,  

             _count_max, _growth_max, _AUC_count, 
             _count_mean, 
             ]
    func_names = ['max count', 'max growth (platelets/s)', 'count AUC', 
                  'mean count', 'time max count', 'mean growth (platelets/s)',

                  'max size (um)', 'max growth (um/s)', 
                  'size AUC (um^2)', 'mean size (um)', 
                  
                  'max density (platelets/um^2)', 'max contraction (platelets/um^2/s)',
                  'density AUC (platelets/um^4)', 'mean density (platelets/um^2)', 
                  ]
    
    apply_cols = ['platelet count', 'growth (platelets/s)', 'platelet count', 
                  'platelet count', 'time peak count', 'growth (platelets/s)',

                  'size (um)', 'growth (um/s)', 
                  'size (um)', 'size (um)', 
                  
                  'density (platelets/um^2)', 'contraction (platelets/um^2/s)', 
                  'density (platelets/um^2)', 'density (platelets/um^2)', ]
    out = groupby_apply_all(summary_data, apply_cols, funcs, func_names, groupby)
    summary_data_g = summary_data[summary_data['time (s)'] < summary_data['time peak count']]
    out_g = groupby_apply_all(summary_data_g, apply_cols, funcs, func_names, groupby)
    summary_data_c = summary_data[summary_data['time (s)'] > summary_data['time peak count']]
    out_c = groupby_apply_all(summary_data_c, apply_cols, funcs, func_names, groupby)
    for func in func_names:
        out = centile_of_score_grouped(out, func, insideout)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data.csv')
    out.to_csv(sp)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data-growth.csv')
    out_g.to_csv(sp)
    sp = os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_centile-data-consolidation.csv')
    out_c.to_csv(sp)
    return out, out_g, out_c, summary_data, peaks_data


def centile_of_score_grouped(out, col, insideout):
    n = col + ' pcnt'
    if insideout:
        gb = ['treatment', 'inside injury']
    else:
        gb = 'treatment'
    for k, grp in out.groupby(gb):
        def func(val):
            group_data = grp[col].values
            pcnt = stats.percentileofscore(group_data, val)
            return pcnt
        vals = grp[col].apply(func)
        idx = grp.index.values
        out.loc[idx, n] = vals
    return out




def groupby_apply_all(df, apply_cols, funcs, func_names, groupby):
    '''
    For each group in the groupby (e.g., group = one injury @ one timepoint), 
    apply a list of functions (func). Each function should return a single value from 
    the group, which will added to the function's column (func_names). Each group 
    will contribute a single row in the output dataframe.

    - clunky with a for loop but perfectly reasonable for low numbers of groups
    - probably dont use when grouping by injury X particle ID (platelet tracking ID)
    '''
    t = time()
    print(f'grouping by {groupby}')
    print(f'applying functions to obtain {func_names}')
    out = {col : [] for col in groupby} 
    for n in func_names:
        out[n] = []
    apply_col_u = list(set(apply_cols))
    for col in apply_col_u:
        out[col] = []
    for k, grp in df.groupby(groupby):
        for i, col in enumerate(groupby):
            out[col].append(k[i])
        for col in apply_col_u:
            out[col].append(grp[col].mean())
        for func, name, col in zip(funcs, func_names, apply_cols):
            res = func(grp, col) # res is a scalar
            out[name].append(res)
    out = pd.DataFrame(out)
    t = time() - t
    print(f'Took {t} seconds')
    return out


# -------------------
# Angle analysis data
# -------------------

def angle_binned_outside_injury_phase_data(
        df, 
        peaks,
        save_path, 
        only_large=True, 
        n_bins=40, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        variables=None, 
        funcs=None,
        curry_with=None,
        bin_by_angle_recruited=False, 
        first_three_frames=False
    ):
    '''
    Get a data frame with experiment x phase x phi angle data for only outside
    injury. This will help you to generate a lineplot showing how a measure
    varies with the angle from the midpoint when looking at a thrombus 
    lengthwise (anterior to posterior). Anterior of the mid point, platelets 
    have an angle of > 0 <= 90 degrees. Posterior of the mid point, platelets
    have and angle of >= 90 < 0 degrees. 

    NB: df must have dist_c 
    '''
    # add angle bins
    df = df[df['nrtracks'] > 1]
    if first_three_frames:
        df = df[df['tracknr'] < 4]
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    if not bin_by_angle_recruited:
        _bin_var(df, 'cyl_azimuthal', 90, -90, n_bins, 'angle from midpoint (degrees)')
    else:
        print(df.head())
        _bin_var_by_first_track(df, 'cyl_azimuthal', 90, -90, n_bins, 'angle from midpoint (degrees)')
    # get variables, funcs, and curry_with if not supplied
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=bin_by_angle_recruited)
    # only outside injury
    df = df[df['inside_injury'] == False]
    # get only large for MIPS and DMSO if true
    df = _only_large(only_large, df)
    # results
    gb = ['path', 'treatment', 'phase', 'angle from midpoint (degrees)']
    data = _result_dict(gb, variables)
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        sdf = _time_from_peak_sml_df(tx, ctl, peaks, df)
        # add g/c
        sdf['phase'] = sdf['time from peak count'].apply(_growth_consol)
        for k, grp in sdf.groupby(gb):
            _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


def _generate_var_func_curry(by_first_frame=False):
    if not by_first_frame:
        variables = ('platelet count', 'platelet density gain (um^-3)', 
                   'platelet average density (um^-3)', 'frame average density (um^-3)',
                   'average platelet stability', 'recruitment', 
                   'P(recruited < 15 s)', 'total sliding (um)', 
                   'average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
                   'average contraction in frame (um s^-1)', 'average platelet corrected calcium',
                   'average frame corrected calcium', 'shedding', 
                   'average platelet tracking time (s)', 'P(< 15s)', 
                   'average net platelet loss (/min)', 'average greatest net platelet loss (/min)', 
                   'average platelet elongation', 'total change in velocity (um/s)', 
                   'average platelet y velocity (um/s)', 'P(> 60s)', 
                   'P(recruited > 60 s)', 'platelets from other regions', 
                   'number lost')
        funcs = (count, densification, 
               platelet_mean_of_var, frame_mean_of_var,
               platelet_mean_of_var, recruitment_phase,
                 p_recruited_lt15, total_sliding_phase, 
                 average_sliding_phase, platelet_mean_of_var, 
                 frame_mean_of_var, platelet_mean_of_var, 
                 frame_mean_of_var, shedding_phase, 
                 platelet_average_tracking_time, p_lt15_phase, 
                 frame_average_net_platelet_loss, average_greatest_net_platelet_loss, 
                 platelet_mean_of_var, cumulative_var, 
                 platelet_mean_of_var, p_gt60_phase, 
                 p_recruited_gt60, n_platelets_from_other, 
                 number_lost)
        curry_with = (None, None, 
                    'nb_density_15', 'nb_density_15',
                   'stab', None, 
                   None, None, 
                   None, 'cont', 
                   'cont', 'ca_corr',
                   'ca_corr', None, 
                   None, None, 
                   None, None, 
                   'elong', 'dv', 
                   'dvy', None, 
                   None, None, 
                   None)
    else:
        variables = ('platelet count', 'platelet density gain (um^-3)', 
                   'platelet average density (um^-3)', 'frame average density (um^-3)',
                   'average platelet instability', 'recruitment', 
                   'P(recruited < 15 s)', 'total sliding (um)', 
                   'average platelet sliding (um)', 'average platelet contraction (um s^-1)', 
                   'average contraction in frame (um s^-1)', 'average platelet corrected calcium',
                   'average frame corrected calcium', 'shedding', 
                   'average platelet tracking time (s)', 'initial corrected calcium', 
                   'initial platlet density (um^-3)', 'initial platelet instability', 
                   'angle at time shed', 'y coordinate at time shed', 
                   'distance from centre at time shed', 'intial platelet density gain (um^-3)', 
                   'P(< 15s)', 'initial platelet velocity change (um/s)', 
                   'average net platelet loss (/min)', 'average greatest net platelet loss (/min)', 
                   'average platelet elongation', 'total change in velocity (um/s)', 
                   'average platelet y velocity (um/s)', 'P(> 60s)', 
                   'P(recruited > 60 s)', 'platelets from other regions', 
                   'number lost')
        funcs = (count, densification, 
               platelet_mean_of_var, frame_mean_of_var,
               platelet_mean_of_var, recruitment_phase,
                 p_recruited_lt15, total_sliding_phase, 
                 average_sliding_phase, platelet_mean_of_var, 
                 frame_mean_of_var, platelet_mean_of_var, 
                 frame_mean_of_var, shedding_phase, 
                 platelet_average_tracking_time, var_for_first_3_frames, 
                 var_for_first_3_frames, var_for_first_3_frames, 
                 where_shed, where_shed, 
                 where_shed, initial_platelet_densification, 
                 p_lt15_phase, initial_platelet_velocity_change, 
                 frame_average_net_platelet_loss, average_greatest_net_platelet_loss, 
                 platelet_mean_of_var, cumulative_var, 
                 platelet_mean_of_var, p_gt60_phase, 
                 p_recruited_gt60, n_platelets_from_other, 
                 number_lost)
        curry_with = (None, None, 
                    'nb_density_15', 'nb_density_15',
                   'stab', None, 
                   None, None, 
                   None, 'cont', 
                   'cont', 'ca_corr',
                   'ca_corr', None, 
                   None, 'ca_corr', 
                   'nb_density_15', 'stab', 
                   'cyl_azimuthal', 'ys', 
                   'dist_c', None, 
                   None, None, 
                   None, None, 
                   'elong', 'dv', 
                   'dvy', None, 
                   None, None, 
                   None)
    r = (variables, funcs, curry_with)
    return r


def _bin_var(df, var, ub, lb, n_bins, bin_name):
    u_bins = np.linspace(lb, ub, n_bins)[:-1]
    l_bins = np.linspace(lb, ub, n_bins)[1:]
    bin_func = _value_bin(u_bins, l_bins)  
    vals = df[var].apply(bin_func)  
    df[bin_name] = vals


def _bin_var_by_first_track(df, var, ub, lb, n_bins, bin_name):
    u_bins = np.linspace(lb, ub, n_bins)[:-1]
    l_bins = np.linspace(lb, ub, n_bins)[1:]
    bin_func = _first_frame_bin(df, u_bins, l_bins, var, bin_name)
    df[var].groupby(['path', 'particle']).apply(bin_func)  


@curry
def _value_bin(u_bins, l_bins, val):
    for lb, ub in zip(u_bins, l_bins):
        if val >= lb and val < ub:
            b = (lb + ub) / 2
            return b


@curry
def _first_frame_bin(df, u_bins, l_bins, bin_col, bin_name, grp):
    '''Applied to groupby path & particle'''
    ff = grp[grp['tracknr'] == 1]
    val = ff[bin_col].values[0]
    n_frames = len(grp)
    idxs = grp.index.values
    for lb, ub in zip(u_bins, l_bins):
        if val >= lb and val < ub:
            b = (lb + ub) / 2
            df.loc[idxs, bin_name] = [b, ] * n_frames


def _data_for_group(gb, variables, funcs, curry_with, data, k, grp):
    for i, v in enumerate(variables):
        func = funcs[i]
        if curry_with[i] is not None:
            func = func(curry_with[i])
        val = func(grp)
        data[v].append(val) 
    for i, v in enumerate(gb):
        data[v].append(k[i])


def _result_dict(gb, variables):
    data = {}
    for v in gb:
        data[v] = []
    for v in variables:
        data[v] = []
    return data

def _time_from_peak_sml_df(tx, ctl, peaks, df):
    ttp = peaks[peaks['treatment'] == ctl]['time peak count'].mean()
    sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
    sdf['time from peak count'] = sdf['time (s)'] - ttp
    return sdf


# --------------------------
# Inside injury summary data
# --------------------------


def inside_and_outside_injury_counts_density(
        df, 
        peaks_data, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), ):
    '''
    Get data table of platelet counts and density over time for both
    inside and outside injury. NB: the time variable is zeroed at 
    the control's mean 'time to peak count' (time called 'time from peak count').
    Make sure you add inside_injury before you run this. 

    Parameters
    ----------
    df: pd.DataFrame
        Platelet observation data frame of the type manipulated by variables module
    peaks_data: pd.DataFrame
        Peaks output from rolling_counts_and_growth_plus_peaks() [above]
    save_path: str
        Path to which to save output. The path will be used to generate two files, 
        one ending in '_peaks.csv' and the other ending in '_rolling-counts.csv'. 
    treatments: tuple of str
        Category names of the treatment conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name
    controls: tuple of str
        Category names of the control conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name

    Returns
    -------
    data: pd.DataFrame
    '''
    data = {
        'path' : [], 
        'inside injury' : [], 
        'treatment' : [],
        'time from peak count' : [],
        'platelet count' : [], 
        'platelet density um^-3' : []
    }
    df = df[df['nrtracks'] > 1]
    ldf = df.head()
    ldf.to_csv(os.path.join(Path(save_path).parents[0], 'insideout_debugging.csv'))
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        ttp = peaks_data[peaks_data['treatment'] == ctl]['time peak count'].mean()
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        sdf['time from peak count'] = sdf['time (s)'] - ttp
        for k, grp in sdf.groupby(['path', 'time from peak count', 'treatment', 'inside_injury']):
            data['path'].append(k[0])
            data['time from peak count'].append(k[1])
            data['treatment'].append(k[2])
            data['inside injury'].append(k[3])
            data['platelet count'].append(len(pd.unique(grp.particle)))
            data['platelet density um^-3'].append(np.nanmean(grp['nb_density_15'].values))
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data



def inside_and_outside_injury_counts_density_thrombus_size(
        df, 
        peaks_data, 
        save_path,
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), ):
    '''
    Get data table of platelet counts and density over time for both
    inside and outside injury in both small and large thrombi. 
    NB: the time variable is zeroed at the control's mean 'time to peak count' 
    (time called 'time from peak count'). 
    
    Make sure you add inside_injury and size vars before you run this. 

    Parameters
    ----------
    df: pd.DataFrame
        Platelet observation data frame of the type manipulated by variables module
    peaks_data: pd.DataFrame
        Peaks output from rolling_counts_and_growth_plus_peaks() [above]
    save_path: str
        Path to which to save output. The path will be used to generate two files, 
        one ending in '_peaks.csv' and the other ending in '_rolling-counts.csv'. 
    treatments: tuple of str
        Category names of the treatment conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name
    controls: tuple of str
        Category names of the control conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name

    Returns
    -------
    data: pd.DataFrame
    '''
    data = {
        'path' : [], 
        'inside injury' : [], 
        'size' : [],
        'treatment' : [],
        'time from peak count' : [],
        'platelet count' : [], 
        'platelet density um^-3' : []
    }
    df = df[df['nrtracks'] > 1]
    ldf = df.head()
    ldf.to_csv(os.path.join(Path(save_path).parents[0], 'insideout_debugging.csv'))
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        ttp = peaks_data[peaks_data['treatment'] == ctl]['time peak count'].mean()
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        sdf['time from peak count'] = sdf['time (s)'] - ttp
        for k, grp in sdf.groupby(['path', 'time from peak count', 'treatment', 'inside_injury', 'size']):
            data['path'].append(k[0])
            data['time from peak count'].append(k[1])
            data['treatment'].append(k[2])
            data['inside injury'].append(k[3])
            data['size'].append(k[4])
            data['platelet count'].append(len(pd.unique(grp.particle)))
            data['platelet density um^-3'].append(np.nanmean(grp['nb_density_15'].values))
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data



def experiment_phase_insideout_data(
        df, 
        peaks_data,
        save_path, 
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True
        ):
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    df = add_region_category(df)
    for k, grp in df.groupby(['treatment', 'region']):
        print(k)
        print(pd.unique(grp.path))
    df = df[df['nrtracks'] > 1]
    _add_time_to_peak(df, peaks_data, treatments, controls)
    df['phase'] = df['time from peak count (s)'].apply(_growth_consol)
    regions = pd.unique(df['region'])
    data = {
        'path' : [], 
        'treatment' : [], 
        'phase' : [], 
        'inside_injury' : []
    }
    for v in variables:
        data[v] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    gb = ['path', 'treatment', 'phase', 'inside_injury']
    for k, grp in df.groupby(gb):
        _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data

# --------------------------
# Regions-based summary data
# --------------------------

def experiment_time_region_data(
        df, 
        save_path,
        treatements=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        exp_col='path',
        time_col='time (s)',
        track_lim=1,
        cat_cols=('treatment', ), #, 'size'), 
        only_large=True, 
        size=False
        ):
    '''
    Get a data table describing regions over time (time x region x path). 
    Use basic.add_basic_variables_to_files to ensure all necessary vars are there
    
    Descriptive variables included are: 
        - Platelet count
        - Platelet density (um^-3)
        - thrombus edge distance (um)
        - mean stability
        - mean tracking time (s)
        - sliding (ums^-1)
        - proportion < 15 s
        - proportion > 60 s
        - tracking time IQR (s)

    If using minutes or hsec as time_col also includes:
        - recruitment (s^-1)
        - shedding (s^-1)
        - proportion shed < 15 s
        - proportion shed > 60 s
        - proportion recruited < 15 s
        - proportion recruited > 60 s

    Parameters
    ----------
    df: pd.DataFrame
        Platelet observation data frame (of the type manupulated by variables module)
    save_path: str
        Path to which to save the output (ending in .csv)
    treatments: tuple of str
        Category names of the treatment conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name
    controls: tuple of str
        Category names of the control conditions in the 'treatment' var. 
        As assignmed by basic.get_treatment_name
    exp_col: str
        name of the variable with unique thrombus ID, usually path
    time_col: str
        one of 'time (s)', 'minute', or 'hsec' as defined in the basic module
    cat_cols: tuple
        other columns to collect values for. Takes only the first value in group. 
        The first value in other_cols will be used as the hue for sns.lineplot. 
    only_large: bool
        Use only the large injuries for MIPS and DMSO (MIPS)
    size: bool
        Add additional dimension to data: experiment x time x region x size. 
        One might do this to get vars for statistics. 
    
    Returns
    -------
    data : pd.Dataframe
    
    '''
    var_dict = generate_var_dict(time_col)
    df = _only_large(only_large, df)
    names = list(var_dict.keys())
    funcs = [var_dict[n][0] for n in names]
    curry_with = [var_dict[n][1] for n in names]
    df = df[df['nrtracks'] > track_lim]
    df = add_region_category(df)
    data = []
    for i, func in enumerate(funcs):
        n = names[i]
        print(f'Getting values for {n}...')
        cw = curry_with[i]
        if cw is not None:
            func = func(*cw)
        result = loop_over_exp_region(df, exp_col,time_col, n, func, cat_cols, size)
        result = result.set_index([exp_col, time_col, 'region'], drop=True)
        data.append(result)
    data = pd.concat(data, axis=1)
    data = data.reset_index(drop=False)
    tx = data[cat_cols[0]].values
    if 'Unnamed: 0' in data.columns.values:
        data = data.drop(columns=['Unnamed: 0', ])
    data = data.drop(columns=[cat_cols[0], ])
    data[cat_cols[0]] = tx[:, 0]
    # percentage data
    for n in names:
        nn = f'{n} pcnt'
        data[nn] = [None, ] * len(data)
    nits = len(treatements) * len(pd.unique(data[time_col])) * len(pd.unique(data['region'])) * len(names)
    print('Getting percent of vehicle scores...')
    with tqdm(total=nits) as progress:
        for i, tx in enumerate(treatements):
            v = controls[i]
            tx_df = data[data[cat_cols[0]] == tx]
            for t, g in tx_df.groupby([time_col, 'region']):
                idxs = g.index.values
                v_df = data[(data[cat_cols[0]] == v) & (data[time_col] == t[0]) & (data['region'] == t[1])]
                v_means = {n : v_df[n].mean() for n in names}
                for n in names:
                    nn = f'{n} pcnt'
                    pcnt = g[n].values / v_means[n] * 100
                    data.loc[idxs, nn] = pcnt
                    progress.update(1)
    #debugging_func(data)
    data.to_csv(save_path)
    return data


def loop_over_exp_region(df, exp_col, time_col, val_col, val_func, other_cols, size):
    if isinstance(other_cols, str):
        other_cols = [other_cols, ]
    out = {
        exp_col : [],   
        val_col : [], 
        time_col : [], 
        'region' : []           
    }
    for oc in other_cols:
        out[oc] = []
    nits = len(pd.unique(df[exp_col])) * len(pd.unique(df[time_col])) * len(pd.unique(df['region']))
    if size:
        gb = [exp_col, time_col, 'region', 'size']
    else:
        gb = [exp_col, time_col, 'region']
    with tqdm(total=nits) as progress:
        for k, g in df.groupby(gb): # would have done groupby apply, but I don't care
            val = val_func(g)
            out[exp_col].append(k[0])
            out[val_col].append(val)
            out[time_col].append(k[1])
            out['region'].append(k[2])
            for oc in other_cols:
                out[oc].append(g[oc].values[0]) 
                # can add functionality here (mean) with optional extra arg
            progress.update(1)
    out = pd.DataFrame(out)
    return out


def generate_var_dict(ttype):
    if ttype == 'minute' or ttype == 'hsec':
        var_dict = {
            'platelet count' : [count, None], 
            'platelet density (um^-3)' : [density, ['nb_density_15', ]], 
            'thrombus edge distance (um)' : [outer_edge, ['rho', (90, 98)]], 
            'recruitment (s^-1)' : [recruitment, None], 
            'shedding (s^-1)' : [shedding, None], 
            'mean stability' : [stability, None], 
            'mean tracking time (s)' : [tracking_time, None], 
            'sliding (ums^-1)' : [sliding, None], 
            'proportion < 15 s' : [p_lt15s, None], 
            'proportion > 60 s' : [p_gt60s, None], 
            'tracking time IQR (s)' : [tracking_time_IQR, None], 
            'proportion shed < 15 s' : [p_shed_lt15, None], 
            'proportion shed > 60 s' : [p_shed_gt60, None], 
            'proportion recruited < 15 s' : [p_recruited_lt15, None], 
            'proportion recruited > 60 s' : [p_recruited_gt60, None]
        }
    elif ttype == 'time (s)':
        var_dict = {
            'platelet count' : [count, None], 
            'platelet density (um^-3)' : [density, ['nb_density_15', ]], 
            'thrombus edge distance (um)' : [outer_edge, ['rho', (90, 98)]], 
            'mean stability' : [stability, None], 
            'mean tracking time (s)' : [tracking_time, None], 
            'sliding (ums^-1)' : [sliding, None], 
            'proportion < 15 s' : [p_lt15s, None], 
            'proportion > 60 s' : [p_gt60s, None], 
            'tracking time IQR (s)' : [tracking_time_IQR, None], 
        }
    return var_dict


def region_parallel_coordinate_data(
        df, 
        peaks_data,
        save_path, 
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True
        ):
    df = add_region_category(df)
    df = df[df['nrtracks'] > 1]
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    _add_time_to_peak(df, peaks_data, treatments, controls)
    df['phase'] = df['time from peak count (s)'].apply(_growth_consol)
    regions = pd.unique(df['region'])
    data = {
        'path' : [], 
        'treatment' : [], 
        'phase' : []
    }
    for r in regions:
        for v in variables:
            n = f'{r}: {v}'
            data[n] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    #_apply_to_group = apply_to_group(variables, funcs, curry_with, data)
    c = 0
    for k, grp in df.groupby(['path', 'treatment', 'phase']):
        #_apply_to_group(grp)
        #print(f'df loop: {c}')
        _PC_for_group(variables, funcs, curry_with, data, k, grp, regions)
        c += 1
        
    #out = df.groupby(['path', 'region', 'phase']).apply(_apply_to_group)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data

#@curry
def _PC_for_group(variables, funcs, curry_with, data, k, grp, regions):
    #paths = grp['path'].values[0]
    c = 0
    for i, v in enumerate(variables):
        for r in regions:
            rgrp = grp[grp['region'] == r]
            func = funcs[i]
            if curry_with[i] is not None:
                func = func(curry_with[i])
            val = func(rgrp)
            n = f'{r}: {v}'
            data[n].append(val) # 145 - too few of these
            c += 1
    #print(f'var loops: {c}')
    c = 0
    data['path'].append(k[0]) # 580
    data['treatment'].append(k[1])
    data['phase'].append(k[2])
    c += 1
    #print(f'cat loops: {c}')


def _growth_consol(val):
    if val > 0:
        return 'consolidation'
    else:
        return 'growth'

def _add_time_to_peak(df, peaks, treatments, controls):
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        ctl_peaks = peaks[peaks['treatment'] == ctl]
        peak = ctl_peaks['time peak count'].mean()
        tfp = sdf['time (s)'] - peak
        idx = sdf.index.values
        df.loc[idx, 'time from peak count (s)'] = tfp


def _add_time_to_max(df, max_data, treatments, controls):
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        sdf = pd.concat([df[df['treatment'] == tx], df[df['treatment'] == ctl]])
        ctl_max = max_data[max_data['treatment'] == ctl]
        peak = ctl_max['time_max_count'].mean()
        tfp = sdf['time (s)'] - peak
        idx = sdf.index.values
        df.loc[idx, 'time from peak count (s)'] = tfp


def _only_large(only_large, df):
    if only_large:
        mips = df[df['treatment'] == 'MIPS']
        print(mips.columns.values)
        dmso = df[df['treatment'] == 'DMSO (MIPS)']
        other = df[(df['treatment'] != 'MIPS') & (df['treatment'] != 'DMSO (MIPS)')]
        mips = mips[mips['size'] == 'large']
        dmso = dmso[dmso['size'] == 'large']
        df = pd.concat([mips, dmso, other]).reset_index(drop=True)
    return df


def experiment_region_phase_data(
        df, 
        peaks_data,
        save_path, 
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True, 
        phase_type='max'
        ):
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    cols = ['pid', 'path', 'frame', 'x_s', 'ys', 'zs', 'treatment', 'stab', 'nba_d_5', 
            'dvx', 'dvy', 'dvz', 'dv', 'particle', 'cont', 'tracknr', 'nrtracks', 'tracked', 
            'inside_injury', 'ca_corr', 'nb_density_15', 'rho', 'theta', 
            'phi', 'cyl_r', 'time (s)',  'terminating', 'sliding (ums^-1)', 
            'total time tracked (s)', 'tracking time (s)', 'size','elongation', 'dist_c', 'elong']
    df = df[cols]
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    df = add_region_category(df)
    df = df[df['nrtracks'] > 1]
    for k, grp in df.groupby(['treatment', 'region']):
        print(k)
        print(pd.unique(grp.path))
    if phase_type == 'peak':
        _add_time_to_peak(df, peaks_data, treatments, controls)
    elif phase_type == 'max':
        _add_time_to_max(df, peaks_data, treatments, controls)
    df['phase'] = df['time from peak count (s)'].apply(_growth_consol)
    regions = pd.unique(df['region'])
    data = {
        'path' : [], 
        'treatment' : [], 
        'phase' : [], 
        'region' : []
    }
    for v in variables:
        data[v] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    c = 0
    for k, grp in df.groupby(['path', 'treatment', 'phase', 'region']):
        _ERP_for_group(variables, funcs, curry_with, data, k, grp, regions)
        c += 1
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


def _ERP_for_group(variables, funcs, curry_with, data, k, grp, regions):
    for i, v in enumerate(variables):
        func = funcs[i]
        if curry_with[i] is not None:
            func = func(curry_with[i])
        val = func(grp)
        data[v].append(val) 
    data['path'].append(k[0]) 
    data['treatment'].append(k[1])
    data['phase'].append(k[2])
    data['region'].append(k[3])




def experiment_region_epoch_data(
        df, 
        save_path, 
        epochs=((0, 100), (100, 300), (300, 600)), 
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True, 
        ):
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    cols = ['pid', 'path', 'frame', 'x_s', 'ys', 'zs', 'treatment', 'stab', 'nba_d_5', 
            'dvx', 'dvy', 'dvz', 'dv', 'particle', 'cont', 'tracknr', 'nrtracks', 'tracked', 
            'inside_injury', 'ca_corr', 'nb_density_15', 'rho', 'theta', 
            'phi', 'cyl_r', 'time (s)',  'terminating', 'sliding (ums^-1)', 
            'total time tracked (s)', 'tracking time (s)', 'size','elongation', 'dist_c', 'elong']
    df = df[cols]
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    df = add_region_category(df)
    df = df[df['nrtracks'] > 1]
    for k, grp in df.groupby(['treatment', 'region']):
        print(k)
        print(pd.unique(grp.path))
    epoch_binnning = _epoch(epochs)
    df['epoch'] = df['time (s)'].apply(epoch_binnning)
    regions = pd.unique(df['region'])
    data = {
        'path' : [], 
        'treatment' : [], 
        'epoch' : [], 
        'region' : []
    }
    for v in variables:
        data[v] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    gb = ['path', 'treatment', 'epoch', 'region']
    for k, grp in df.groupby(gb):
        _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data

@curry
def _epoch(epochs, t):
    for l, u in epochs:
        if t >= l and t < u:
            return f'{l}-{u}'

# ------------------------
# Cylindrical regions data
# ------------------------

def cylrad_binned_region_phase_data(
        df, 
        peaks,
        save_path, 
        only_large=True, 
        n_bins=50, 
        treatments=('MIPS', 'SQ', 'cangrelor'), 
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), 
        variables=None, 
        funcs=None,
        curry_with=None,
    ):
    '''
    NB: df must have cyl_radial
    '''
    df = df[df['tracknr'] > 1]
    # add angle bins
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    if 'cyl_radial' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    _bin_var(df, 'cyl_radial', 90, -90, n_bins, 'XY distance from midpoint (um)')
    # get proper quadrants
    if 'quadrant' not in df.columns.values:
        from plateletanalysis.variables.basic import quadrant_var
        df = quadrant_var(df)
    # get only large for MIPS and DMSO if true
    df = _only_large(only_large, df)
    # TODO: only include ant, lat, and post if necessary
    gb = ['path', 'treatment', 'phase', 'quadrant', 'XY distance from midpoint (um)']
    data = _result_dict(gb, variables)
    for i, ctl in enumerate(controls):
        tx = treatments[i]
        sdf = _time_from_peak_sml_df(tx, ctl, peaks, df)
        # add g/c
        sdf['phase'] = df['time from peak count'].apply(_growth_consol)
        for k, grp in sdf.groupby(gb):
            _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data



def regions_heatmap_data(df, save_path, var_dict, group='MIPS', pcnt_treatments=('MIPS', 'SQ', 'cangrelor'), 
        pcnt_controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), count_var='platelet count', ordered=False):
    '''
    Uses data from experiment_region_phase
    '''
    vars_list = list(var_dict.keys())
    df = pcnt_veh('region', vars_list, pcnt_treatments, pcnt_controls, df)
    df = df[df['treatment'] == group]
    sizes = {
        'path' : [], 
        'count' : [], 
    }
    for k, grp in df.groupby('path'):
        count = grp[count_var].sum()
        sizes['count'].append(count)
        sizes['path'].append(k)
    sizes = pd.DataFrame(sizes)
    sizes = sizes.sort_values('count')
    sizes = sizes.reset_index(drop=True)
    sizes = sizes.reset_index().rename(columns={'index' : 'rank'})
    assign_rank = _assign_rank(sizes)
    df['rank'] = df['path'].apply(assign_rank)
    rename = {}
    names = []
    for k in var_dict.keys():
        old = f'{k} (% vehicle)'
        new = var_dict[k]
        rename[old] = new
        if k == new:
            df = df.rename(columns={k : f'{k} raw'})
        names.append(new)
    df = df.rename(columns=rename)
    df['region x rank'] = df['region'] + ': ' + df['rank']
    names = ['region x rank', ] + names
    new_df = df[names]
    max_rank = sizes['rank'].max()
    ranks = list(range(max_rank))
    regions = ['center', 'anterior', 'lateral', 'posterior']
    order = []
    for r in regions:
        for i in ranks:
            order.append(f'{r}: {i}')
    new_df = new_df.set_index('region x rank')
    new_df = new_df.loc[order, :]
    rows = [row for row in order if row.startswith('lateral')]
    if ordered:
        cols = new_df.columns.values
        scores = [new_df.loc[rows, col].values.mean() for col in cols]
        indexs = np.argsort(scores)
        column_order = cols[indexs]
        new_df = new_df[column_order]
    new_df = new_df.reset_index(drop=False)
    new_df.to_csv(save_path)
    return new_df


@curry
def _assign_rank(ranks_df, path):
    rank = ranks_df[ranks_df['path'] == path]['rank']
    return str(rank.values[0])



def pcnt_veh(gb, vars, treatments, controls, df):
    cols = [f'{var} (% vehicle)' for var in vars]
    for tx, ctl in zip(treatments, controls):
        ctl_df = df[df['treatment'] == ctl].copy()
        sdf = pd.concat([df[df['treatment'] == tx], ctl_df]).copy()
        for var in vars:
            min_val = sdf[var].min()
            if min_val < 0:
                snv = sdf[var] - min_val
                sdf[var] = snv
                cnv = ctl_df[var] - min_val
                ctl_df[var] = cnv
        for k, grp in sdf.groupby(gb): 
            idxs = grp.index.values
            for var, n in zip(vars, cols):
                orig = grp[var].values
                if isinstance(k, tuple):
                    kdf = ctl_df.copy()
                    for gbcol, k_val in zip(gb, k):
                        kdf = kdf[kdf[gbcol] == k_val]
                    var_veh_mean = kdf[var].mean()
                else:
                    var_veh_mean = ctl_df[ctl_df[gb] == k][var].mean()
                vals = orig / var_veh_mean * 100
                df.loc[idxs, n] = vals
                if vals.min() < 0:
                    pass
    return df


# -------------
# Regions ISO A
# -------------

def quadrant_isoA_phase_data(
        df, 
        peaks_data,
        save_path, 
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True, 
        include_untracked=False, 
        phase_type='max'
        ):
    cols = ['pid', 'path', 'frame', 'x_s', 'ys', 'zs', 'treatment', 'stab', 'nba_d_5', 
            'dvx', 'dvy', 'dvz', 'dv', 'particle', 'cont', 'tracknr', 'nrtracks', 'tracked', 
            'inside_injury', 'ca_corr', 'nb_density_15', 'rho', 'theta', 
            'phi', 'cyl_r', 'time (s)',  'terminating', 'sliding (ums^-1)', 
            'total time tracked (s)', 'tracking time (s)', 'size','elongation', 'dist_c', 'elong']
    df = df[cols]
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    if 'iso_A' not in df.columns.values:
        from plateletanalysis.variables.basic import isoA_var
        df = isoA_var(df)
    if 'quadrant' not in df.columns.values:
        from plateletanalysis.variables.basic import quadrant_var
        df = quadrant_var(df)
    #df = add_region_category(df)
    if not include_untracked:
        df = df[df['nrtracks'] > 1]
    if phase_type == 'peak':
        _add_time_to_peak(df, peaks_data, treatments, controls)
    elif phase_type == 'max':
        _add_time_to_max(df, peaks_data, treatments, controls)
    df['phase'] = df['time from peak count (s)'].apply(_growth_consol)
    _bin_var_str(df, 'iso_A', 70, 10, 12, 'iso_A_bin', 0)
    data = {
        'iso_A_bin' : [],
        'treatment' : [], 
        'phase' : [], 
        'quadrant' : [], 
    }
    for v in variables:
        data[v] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    c = 0
    gb = ['iso_A_bin', 'treatment', 'phase', 'quadrant']
    for k, grp in df.groupby(gb):
        _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
        c += 1
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


def quadrant_isoA_heatmap_data(df, save_path, var_dict, group='MIPS', pcnt_treatments=('MIPS', 'SQ', 'cangrelor'), 
        pcnt_controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'), count_var='platelet count', ordered=False):
    '''
    Uses data from quadrant_isoA_phase_data
    '''
    vars_list = list(var_dict.keys())
    df = pcnt_veh(['quadrant', 'iso_A_bin'], vars_list, pcnt_treatments, pcnt_controls, df)
    df = df[df['treatment'] == group]
    
    
    # rename columns
    rename = {}
    names = []
    for k in var_dict.keys():
        old = f'{k} (% vehicle)'
        new = var_dict[k]
        rename[old] = new
        if k == new:
            df = df.rename(columns={k : f'{k} raw'})
        names.append(new)
    df = df.rename(columns=rename)
    
    df['quadrant x iso_A'] = df['quadrant'] + ': ' + df['iso_A_bin']
    names = ['quadrant x iso_A', ] + names
    new_df = df[names]
    quadrants = ['anterior', 'lateral', 'posterior']
    isoAbins = list(sorted(pd.unique(df['iso_A_bin'])))
    order = []
    for r in quadrants:
        for i in isoAbins:
            order.append(f'{r}: {i}')
    new_df = new_df.set_index('quadrant x iso_A')
    new_df = new_df.loc[order, :]
    if ordered:
        rows = [row for row in order if row.startswith('lateral')]
        cols = new_df.columns.values
        scores = [new_df.loc[rows, col].values.mean() for col in cols]
        indexs = np.argsort(scores)
        column_order = cols[indexs]
        new_df = new_df[column_order]
    new_df = new_df.reset_index(drop=False)
    new_df.to_csv(save_path)
    return new_df 


def _bin_var_str(df, var, ub, lb, n_bins, bin_name, dec):
    u_bins = np.linspace(lb, ub, n_bins)[:-1]
    l_bins = np.linspace(lb, ub, n_bins)[1:]
    bin_func = _value_bin_str(dec, u_bins, l_bins)  
    vals = df[var].apply(bin_func)  
    df[bin_name] = vals


@curry
def _value_bin_str(dec, u_bins, l_bins, val):
    for lb, ub in zip(u_bins, l_bins):
        if val >= lb and val < ub:
            r_lb = np.round(lb, decimals=dec)
            r_ub = np.round(ub, decimals=dec)
            b = f'{r_lb}-{r_ub}'
            return b



def experiment_quadrant_isoA_phase_data(
        df, 
        peaks_data,
        save_path, 
        treatments=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        variables=None, 
        funcs=None,
        curry_with=None,
        only_large=True, 
        include_untracked=False, 
        phase_type='max', 
        bin_edges=(25, 45, 65)
        ):
    cols = ['pid', 'path', 'frame', 'x_s', 'ys', 'zs', 'treatment', 'stab', 'nba_d_5', 
            'dvx', 'dvy', 'dvz', 'dv', 'particle', 'cont', 'tracknr', 'nrtracks', 'tracked', 
            'inside_injury', 'ca_corr', 'nb_density_15', 'rho', 'theta', 
            'phi', 'cyl_r', 'time (s)',  'terminating', 'sliding (ums^-1)', 
            'total time tracked (s)', 'tracking time (s)', 'size','elongation', 'dist_c', 'elong']
    df = df[cols]
    if variables is None:
        variables, funcs, curry_with = _generate_var_func_curry(by_first_frame=True)
    if 'cyl_azimuthal' not in df.columns.values:
        from plateletanalysis import cylindrical_coordinates
        df = cylindrical_coordinates(df)
    if 'iso_A' not in df.columns.values:
        from plateletanalysis.variables.basic import isoA_var
        df = isoA_var(df)
    if 'quadrant' not in df.columns.values:
        from plateletanalysis.variables.basic import quadrant_var
        df = quadrant_var(df)
    #df = add_region_category(df)
    if not include_untracked:
        df = df[df['nrtracks'] > 1]
    if phase_type == 'peak':
        _add_time_to_peak(df, peaks_data, treatments, controls)
    elif phase_type == 'max':
        _add_time_to_max(df, peaks_data, treatments, controls)
    df['phase'] = df['time from peak count (s)'].apply(_growth_consol)
    rho_binning = _rho_binning(bin_edges)
    df['rho_bin'] = df['iso_A'].apply(rho_binning)
    data = {
        'path' : [],
        'rho_bin' : [],
        'treatment' : [], 
        'phase' : [], 
        'quadrant' : [], 
    }
    for v in variables:
        data[v] = []
    # only large MIPS and DMSO MIPS
    df = _only_large(only_large, df)
    # get data
    c = 0
    gb = ['path', 'rho_bin', 'treatment', 'phase', 'quadrant']
    for k, grp in df.groupby(gb):
        _data_for_group(gb, variables, funcs, curry_with, data, k, grp)
        c += 1
    data = pd.DataFrame(data)
    data.to_csv(save_path)
    return data


@curry
def _rho_binning(bin_edge, r):
    if r <= bin_edge[0]:
        return f'0-{bin_edge[0]} um'
    elif r > bin_edge[0] and r <= bin_edge[1]:
        return f'{bin_edge[0]}-{bin_edge[1]} um'
    else:
        return f'{bin_edge[1]}-{bin_edge[2]} um'



# --------------------------
# First few frames over time
# --------------------------

def first_3_hsec_binned_data():
    pass


def first_5_trknr_binned_data():
    pass