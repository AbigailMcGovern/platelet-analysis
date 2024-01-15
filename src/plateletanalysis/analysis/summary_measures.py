import pandas as pd
import numpy as np
from scipy import stats
from plateletanalysis.variables.basic import add_region_category
from toolz import curry
from plateletanalysis.variables.measure import quantile_normalise_variables



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
    lt15 = len(pd.unique(df[df['nrtracks'] < 15]['particle']))
    return lt15 / t


def p_gt60_phase(df):
    t = len(pd.unique(df['particle']))
    gt60 = len(pd.unique(df[df['nrtracks'] > 60]['particle']))
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


