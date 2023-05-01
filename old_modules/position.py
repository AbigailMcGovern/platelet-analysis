from toolz import curry
import numpy as np
import pandas as pd
import math as m

from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
from .. import config as cfg
from .transform import spherical_coordinates



def add_scale_free_positional_categories(df, dens_col='nb_density_15', donut_info=None):
    '''
    Add catrgories of position dependent on anterior, lateral, posterior, and surface position. 
    Position is defined accordning to quantile rather than physical coordinates to scale between clots. 
    Positions and identity in output dataframe:
        - surface*: df['surface_or_core'] = 'surface'
        - core*: df['surface_or_core'] = 'core'
        - interior surface*: df['surface_or_core'] = 'interior surface'
        - anterior surface: df['anterior_surface'] = 'True'
        - lateral surface: df['lateral_surface'] = 'True'
        - tail: df['tail'] = 'True'
        - donut: df['inner_donut']= True
    * surface, core, and interior surface are mutually exclusive

    '''
    df = quantile_normalise_dist_sectors(df)
    dens_pcnt = dens_col + '_pcntf'
    if dens_pcnt not in df.columns.values:
        df = quantile_normalise_variables_frame(df, (dens_col, ))
    # add core or surface 'surface_or_core'
    df = add_surface_core(df)
    # add anterior surface
    df = add_anterior_surface(df)
    # add lateral surface and tail
    df = add_lateral_surface_and_tail(df)
    if donut_info is not None:
        # add donut
        df = add_donut_platelets(df, donut_info, dens_col=dens_pcnt, 
                                 dens_thresh=40, z_col='zs', z_thresh_l=8, 
                                 z_thresh_h=56, dist_col='dist_c_pcntf', 
                                 dist_thresh=30)
    return df



def quantile_normalise_dist_sectors(df):
    '''
    Quantile normalise centre distance variable (dist_c) according to 6 sectors
    1) Anterior right - (x > 0, y > y) somtimes thrombi are asymmetrical
    2) Anterior left - (x < 0, y > 0)
    3) Posterior right front - (x > 0, y < 0, > - 45 deg)
    4) Posterior left front - (x < 0, y < 0, > - 45 deg)
    5) Posterior right back - (x > 0, y < 0, < - 45 deg)
    5) Posterior left back - (x < 0, y < 0, < - 45 deg)
    This preserves the surface shape even when thrombi are extremely elongated
    '''
    df = df[df['nrtracks'] > 10]
    if 'phi' not in df.columns.values:
        print('finding phi')
        df['pid'] = range(len(df))
        df = spherical_coordinates(df)
    # anterior
    ant_df = df[df['ys'] > 0]
    # anterior right
    antpx_df = ant_df[ant_df['x_s']>0]
    antpx_df = quantile_normalise_variables_frame(antpx_df, ('dist_c', ))
    # anterior left
    antnx_df = ant_df[ant_df['x_s']<0]
    antnx_df = quantile_normalise_variables_frame(antnx_df, ('dist_c', ))
    # posterior
    pos_df = df[df['ys'] < 0]
    # posterior right
    pospx_df = pos_df[pos_df['x_s'] > 0]
    # posterior right front
    pospxF_df = pos_df[pos_df['phi'] > -0.78539]
    pospxF_df = quantile_normalise_variables_frame(pospxF_df, ('dist_c', ))
    # posterior right back
    pospxB_df = pos_df[pos_df['phi'] < -0.78539]
    pospxB_df = quantile_normalise_variables_frame(pospxB_df, ('dist_c', ))
    # posterior left
    posnx_df = pos_df[pos_df['x_s'] < 0]
    # posterior left front
    posnxF_df = pos_df[pos_df['phi'] > -0.78539]
    posnxF_df = quantile_normalise_variables_frame(posnxF_df, ('dist_c', ))
    # posterior left back
    posnxB_df = pos_df[pos_df['phi'] < -0.78539]
    posnxB_df = quantile_normalise_variables_frame(posnxB_df, ('dist_c', ))
    # concat
    df = pd.concat([antpx_df, antnx_df, pospxF_df, pospxB_df, posnxF_df, posnxB_df])
    df = df.reset_index(drop=True)
    return df



# probably need to add quantile normalised local density (dens_col) and quantile normalised dist_c (dist_col) first 
def add_surface_core(
        df, 
        dens_col='nb_density_15_pcntf', 
        dens_thresh=50, 
        z_col='zs', 
        z_thresh=8, 
        dist_col='dist_c_pcntf', 
        dist_thresh=40
        ):
    files = pd.unique(df['path'])
    df = df.reset_index(drop=True)
    for k, g in df.groupby(['path', ]):
        # sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40)]
        sdf = g[(g[dens_col] < dens_thresh) & (g[z_col] > z_thresh) & (g[dist_col] > dist_thresh)]
        s_idxs = sdf.index.values
        df.loc[s_idxs, 'surface_or_core'] = 'surface'
        del sdf
        cdf = g[g[dens_col] > dens_thresh]
        c_idxs = cdf.index.values
        df.loc[c_idxs, 'surface_or_core'] = 'core'
        del cdf
        idf = g[(g[dens_col] < dens_thresh) & (g[dist_col] < dist_thresh)]
        i_idxs = idf.index.values
        df.loc[i_idxs, 'surface_or_core'] = 'interior surface'
    return df


def add_anterior_surface(df):
    sdf = df[df['surface_or_core'] == 'surface']
    adf = sdf[sdf['ys'] > 0]
    idx = adf.index.values
    df['anterior_surface'] = False
    df.loc[idx, 'anterior_surface'] = True
    return df


def add_lateral_surface_and_tail(df):
    if 'x_s_pcnt' not in df.columns.values:
        df = quantile_normalise_variables(df, ['x_s'])
    sdf = df[df['surface_or_core'] == 'surface']
    ldf = pd.concat([sdf[sdf['x_s_pcnt'] > 25], sdf[sdf['x_s_pcnt'] < 75]])
    idx = ldf.index.values
    df['lateral_surface'] = False
    df.loc[idx, 'lateral_surface'] = True
    del ldf
    tdf = sdf[sdf['ys'] < 0]
    tdf = pd.concat([tdf[tdf['x_s_pcnt'] > 25], tdf[tdf['x_s_pcnt'] < 75]])
    idx = tdf.index.values
    df['tail'] = False
    df.loc[idx, 'tail'] = True
    return df



def add_donut_platelets(
        df, 
        donut_df,
        dens_col='nb_density_15_pcntf', 
        dens_thresh=40, 
        z_col='zs', 
        z_thresh_l=8, 
        z_thresh_h=56, 
        dist_col='dist_c_pcntf', 
        dist_thresh=30
    ):
    df0 = df.set_index(['path', 'particle'])
    df0['inner_donut'] = False
    for k, g in df.groupby(['path', 'treatment']):
        path = k[0]
        if len(donut_df['frame_min']) > 1:
            row = donut_df[donut_df['treatment'] == k[1]]
        else:
            row = donut_df
        frame_min = row['frame_min'][0] # average first peak - 3
        frame_max = row['frame_max'][0] # average first peak + 5
        ddf = g[(g[dens_col] < dens_thresh) & (g[z_col] > z_thresh_l) & \
                (g[z_col] < z_thresh_h) & (g[dist_col] < dist_thresh) & \
                (g['frame'] > frame_min) & (g['frame'] < frame_max)]
        particles = ddf['particle'].values
        idx = zip([path, ] * len(particles), particles)
        df0.loc[idx, 'inner_donut'] = True
    df0 = df0.reset_index(drop=False)
    return df0



def scale_free_positional_categories(df, thresh=50, var='nb_density_15_pcntf', donut_df=None):
    '''
    Add catrgories of position dependent on anterior, lateral, posterior, and surface position. 
    Position is defined accordning to quantile rather than physical coordinates to scale between clots. 
    Positions and identity in output dataframe:
        - surface*: df['surface_or_core'] = 'surface'
        - core*: df['surface_or_core'] = 'core'
        - anterior surface: df['anterior_surface'] = 'True'
        - tail: df['tail'] = 'True'
        - donut: df['inner_donut']= True
    * surface, core, and interior surface are mutually exclusive
    '''
    df = prepare_df(df)
    files = pd.unique(df['path'])
    df0 = df.set_index('pid')
    for f in files:
        fdf = df[df['path'] == f]
        fdf = fdf.reset_index(drop=False)
        df0 = surface_and_core(df0, fdf, var, thresh)
        df0 = tail_platelets(df0, fdf, var, thresh)
        df0 = anterior_surface_platelets(df0, fdf, var, thresh)
        if donut_df is not None:
            if len(donut_df['frame_min']) > 1:
                tx = get_treatment_name(f)
                row = donut_df[donut_df['treatment'] == tx]
            else:
                row = donut_df
            frame_min = row['frame_min'][0] # average first peak - 3
            frame_max = row['frame_max'][0] # average first peak + 5
            min_max = (frame_min, frame_max)
        else:
            min_max = (24, 32)
        df0 = donut_platelets(df0, fdf, var, thresh, min_max)
    return df0



def prepare_df(df):
    '''
    Quantile normalise centre distance variable (dist_c) according to 6 sectors
    1) Anterior right - (x > 0, y > y) somtimes thrombi are asymmetrical
    2) Anterior left - (x < 0, y > 0)
    3) Posterior right front - (x > 0, y < 0, > - 45 deg)
    4) Posterior left front - (x < 0, y < 0, > - 45 deg)
    5) Posterior right back - (x > 0, y < 0, < - 45 deg)
    5) Posterior left back - (x < 0, y < 0, < - 45 deg)
    This preserves the surface shape even when thrombi are extremely elongated
    '''
    df = df[df['nrtracks'] > 10]
    if 'phi' not in df.columns.values:
        print('finding phi')
        df['pid'] = range(len(df))
        df = spherical_coordinates(df)
    if 'nb_density_15_pcntf' not in df.columns.values:
        df = quantile_normalise_variables_frame(df, ('nb_density_15', ))
    # anterior
    ant_df = df[df['ys'] > 0]
    # anterior right
    antpx_df = ant_df[ant_df['x_s']>0]
    antpx_df = quantile_normalise_variables_frame(antpx_df, ('dist_c', ))
    # anterior left
    antnx_df = ant_df[ant_df['x_s']<0]
    antnx_df = quantile_normalise_variables_frame(antnx_df, ('dist_c', ))
    # posterior
    pos_df = df[df['ys'] < 0]
    # posterior right
    pospx_df = pos_df[pos_df['x_s'] > 0]
    # posterior right front
    pospxF_df = pos_df[pos_df['phi'] > -0.78539]
    pospxF_df = quantile_normalise_variables_frame(pospxF_df, ('dist_c', ))
    # posterior right back
    pospxB_df = pos_df[pos_df['phi'] < -0.78539]
    pospxB_df = quantile_normalise_variables_frame(pospxB_df, ('dist_c', ))
    # posterior left
    posnx_df = pos_df[pos_df['x_s'] < 0]
    # posterior left front
    posnxF_df = pos_df[pos_df['phi'] > -0.78539]
    posnxF_df = quantile_normalise_variables_frame(posnxF_df, ('dist_c', ))
    # posterior left back
    posnxB_df = pos_df[pos_df['phi'] < -0.78539]
    posnxB_df = quantile_normalise_variables_frame(posnxB_df, ('dist_c', ))
    # concat
    df = pd.concat([antpx_df, antnx_df, pospxF_df, pospxB_df, posnxF_df, posnxB_df])
    df = df.reset_index(drop=True)
    df['terminating'] = df['tracknr'] == df['nrtracks']
    df = quantile_normalise_variables(df, ['ca_corr'])
    return df


def surface_and_core(df, fdf, var, thresh):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40)]
    sdf = sdf.set_index('pid')
    idxs = sdf.index.values
    df.loc[idxs, 'surface_or_core'] = 'surface'
    cdf = fdf[fdf[var] > thresh]
    cdf = cdf.set_index('pid')
    idxs = cdf.index.values
    df.loc[idxs, 'surface_or_core'] = 'core'
    return df


def tail_platelets(df, fdf, var, thresh):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40) & (fdf['phi'] < -0.78539)]
    sdf = sdf.set_index('pid')
    idxs = sdf.index.values
    df.loc[idxs, 'tail'] = True
    return df


def anterior_surface_platelets(df, fdf, var, thresh):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40) & (fdf['ys'] > 0)]
    sdf = sdf.set_index('pid')
    idxs = sdf.index.values
    df.loc[idxs, 'anterior_surface'] = True
    return df


def donut_platelets(df, fdf, var, thresh, min_max):
    # MIPS = 24-32
    # SQ = 15-26
    # biva = 5-15
    # cang = 4-14
    # saline = 
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] < 30) & (fdf['zs'] < 56) & (fdf['frame'] > min_max[0]) & (fdf['frame'] < min_max[1])]
    particles = sdf['particle'].values
    fdf = fdf.set_index('particle')
    fdf['donut_p'] = False
    fdf.loc[particles, 'donut_p'] = True
    fdf = fdf.reset_index()
    sdf = fdf[fdf['donut_p'] == True]
    sdf = sdf.set_index('pid')
    idxs = sdf.index.values
    df.loc[idxs, 'donut'] = True
    return df



# ------------------------------
# Measurements based on position
# ------------------------------



def count_variables(out, cdf, ndf, idx, column, condition, prefix):
    cdf = cdf[cdf[column] == condition]
    count0 = len(cdf)
    ndf0 = ndf.copy()
    ndf = ndf[ndf[column] == condition]
    count1 = len(ndf)
    tdf = cdf[cdf['terminating'] == True]
    terminating = len(tdf) # lost at the next time point
    particles0 = pd.unique(cdf['particle'])
    particles1 = pd.unique(ndf['particle'])  
    particles2 = pd.unique(ndf0['particle'])  
    if len(particles1) > 0 and len(particles0) > 0:
        gained = [p for p in particles1 if p not in particles0]
        lost = [p for p in particles0 if p not in particles1]
        gained = len(gained)
        lost = len(lost)
        remaining = [p for p in particles0 if p in particles2]
        out = transitions(ndf0, remaining, column, condition, out, idx, prefix)
    else:
        gained = 0
        lost = 0
    count_n = prefix + 'count'
    out.loc[idx, count_n] = count0
    turnover_n = prefix + 'turnover'
    out.loc[idx, turnover_n] = count1 - count0
    turnover_pcnt_n = prefix + 'turnover (%)'
    turnover_pcnt_max_n = prefix + 'turnover (% max)'
    if count0 > 0:
        out.loc[idx, turnover_pcnt_n] = (count1 - count0) / count0 * 100
    else:
        out.loc[idx, turnover_pcnt_n] = 0.
    gained_n = prefix + 'gained'
    out.loc[idx, gained_n] = gained
    gained_pcnt_n = prefix + 'gained (%)'
    if count0 > 0:
        out.loc[idx, gained_pcnt_n] = gained / count0 * 100
    else:
        out.loc[idx, gained_pcnt_n] = 0.
    lost_n = prefix + 'lost'
    out.loc[idx, lost_n] = lost
    lost_pcnt_n = prefix + 'lost (%)'
    if count0 > 0:
        out.loc[idx, lost_pcnt_n] = lost / count0 * 100
    else:
        out.loc[idx, lost_pcnt_n] = 0.
    term_n = prefix + 'terminating'
    out.loc[idx, term_n] = terminating
    term_pcnt_n = prefix + 'terminating (%)'
    if count0 > 0:
        out.loc[idx, term_pcnt_n] = terminating / count0 * 100
    else:
        out.loc[idx, term_pcnt_n] = 0
    return out


    
def transitions(ndf, remaining, col, cond, out, idx, pre):
    res = next_pos(remaining, ndf, cols=('surface_or_core', 'anterior_surface', 'tail', 'donut'))
    for c in res.keys():
        vals = res[c]
        if c == col:
            same = [v for v in vals if v == cond]
            same = len(same)
            same_n = pre + 'stable'
            out.loc[idx, same_n] = same
        if cond == 'surface' and c == 'surface_or_core':
            surf_to_core = [v for v in vals if v == 'core']
            surf_to_core = len(surf_to_core)
            stc_n = pre + 'to core'
            out.loc[idx, stc_n] = surf_to_core
        if cond == 'core' and c == 'surface_or_core':
            core_to_surf = [v for v in vals if v == 'surface']
            core_to_surf = len(core_to_surf)
            cts_n = pre + 'to surface'
            out.loc[idx, cts_n] = core_to_surf
        if cond == 'core' and c == 'tail':
            core_to_tail = [v for v in vals if v == True]
            core_to_tail = len(core_to_tail)
            ctt_n = pre + 'to tail'
            out.loc[idx, ctt_n] = core_to_tail
        if cond == True and col in ('anterior_surface', 'tail') and c == 'surface_or_core':
            _to_core = [v for v in vals if v == 'core']
            _to_core = len(_to_core)
            tc_n = pre + 'to core'
            out.loc[idx, tc_n] = _to_core
        if cond == True and col == 'donut' and c == 'surface_or_core':
            donut_in_core = [v for v in vals if v == 'core']
            donut_in_core = len(donut_in_core)
            dic_n = pre + 'in core'
            out.loc[idx, dic_n] = donut_in_core
    return out



def next_pos(particles, ndf, cols=('surface_or_core', 'anterior_surface', 'tail', 'donut')):
    out = {col : [] for col in cols}
    ndf = ndf.reset_index()
    ndf = ndf.set_index('particle')
    for c in cols:
        nxt = ndf.loc[particles, c].values
        out[c] = nxt
    return out




def get_treatment_name(inh): # need to rename from last run 
    if 'saline' in inh:
        out = 'saline'
    elif 'cang' in inh:
        out = 'cangrelor'
    elif 'veh-mips' in inh:
        out = 'MIPS vehicle'
    elif 'mips' in inh or 'MIPS' in inh:
        out = 'MIPS'
    elif 'sq' in inh:
        out = 'SQ'
    elif 'par4--biva' in inh:
        out = 'PAR4-- bivalirudin'
    elif 'par4--' in inh:
        out = 'PAR4--'
    elif 'biva' in inh:
        out = 'bivalirudin'
    elif 'SalgavDMSO' in inh or 'gavsalDMSO' in inh or 'galsavDMSO' in inh:
        out = 'DMSO (salgav)'
    elif 'Salgav' in inh or 'gavsal' in inh:
        out = 'salgav'
    elif 'DMSO' in inh:
        out = 'DMSO (MIPS)'
    elif 'dmso' in inh:
        out = 'DMSO (SQ)'
    elif 'ctrl' in inh:
        out = 'control'
    else:
        out = inh
    return out