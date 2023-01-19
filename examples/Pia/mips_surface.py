import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from plateletanalysis.variables.measure import quantile_normalise_variables_frame, quantile_normalise_variables
from plateletanalysis.variables.transform import spherical_coordinates
from tqdm import tqdm

# ------------------------
# Add positional variables
# ------------------------


def prepare_df(df):
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



def add_vars(df, thresh=50, var='nb_density_15_pcntf'):
    files = pd.unique(df['path'])
    df0 = df.set_index('pid')
    for f in files:
        fdf = df[df['path'] == f]
        fdf = fdf.reset_index(drop=False)
        df0 = surface_and_core(df0, fdf, var, thresh)
        df0 = tail_platelets(df0, fdf, var, thresh)
        df0 = anterior_surface_platelets(df0, fdf, var, thresh)
        df0 = donut_platelets(df0, fdf, var, thresh, (24, 32))
    return df0



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



# --------------------------------
# Positional information summaries
# --------------------------------


def positional_information_data(paths, save_path):
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        df = prepare_df(df)
        df = add_vars(df)
        dfs.append(df)
    dfs = pd.concat(dfs)
    out = generate_df(dfs)
    out.to_csv(save_path)
    return out



def generate_df(
        df, 
        positional_cols=('surface_or_core', 'surface_or_core', 'anterior_surface', 'tail', 'donut'),
        conditions=('surface', 'core', True, True, True)
        ):
    out = {
        'path' : [], 
        'frame' : [], 
        'time (s)' : [], 
        'treatment' : [], 
    }
    its = 0
    for k, grp in df.groupby(['path', 'frame']):
        p = k[0]
        f = k[1]
        t = f / 0.321764322705706
        tx = get_treatment_name(p)
        out['path'].append(p)
        out['frame'].append(f)
        out['time (s)'].append(t)
        out['treatment'].append(tx)
        its += 1
    out = pd.DataFrame(out)
    with tqdm(desc='Adding averages and count variables', total=its) as progress:
        for k, grp in df.groupby(['path', 'frame']):
            p = k[0]
            f = k[1]
            odf = out[(out['path'] == p) & (out['frame'] == f)]
            idx = odf.index.values
            out = _add_averages(out, grp, idx)
            ndf = df[(df['path'] == p) & (df['frame'] == f + 1)]
            for i, col in enumerate(positional_cols):
                cond = conditions[i]
                if isinstance(cond, str):
                    pre = cond + ' '
                else:
                    pre = col + ' '
                out = _add_averages(out, grp, idx, pre, col, cond)
                out = _add_count_vars(out, grp, ndf, idx, col, cond, pre)
            progress.update(1)
    return out



def _add_count_vars(out, cdf, ndf, idx, column, condition, prefix):
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
        out = get_transitions(ndf0, remaining, column, condition, out, idx, prefix)
    else:
        gained = 0
        lost = 0
    count_n = prefix + 'count'
    out.loc[idx, count_n] = count0
    turnover_n = prefix + 'turnover'
    out.loc[idx, turnover_n] = count1 - count0
    turnover_pcnt_n = prefix + 'turnover (%)'
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


    
def get_transitions(ndf, remaining, col, cond, out, idx, pre):
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
        if cond == True and col == 'anterior_surface' and c == 'surface_or_core':
            asurf_to_core = [v for v in vals if v == 'core']
            asurf_to_core = len(asurf_to_core)
            atc_n = pre + 'to core'
            out.loc[idx, atc_n] = asurf_to_core
    return out



def next_pos(particles, ndf, cols=('surface_or_core', 'anterior_surface', 'tail', 'donut')):
    out = {col : [] for col in cols}
    ndf = ndf.reset_index()
    ndf = ndf.set_index('particle')
    for c in cols:
        nxt = ndf.loc[particles, c].values
        out[c] = nxt
    return out



def _add_averages(
        out, 
        df, 
        idx, 
        prefix='',
        column=None,
        condition=None,
        variables=('dv', 'dvz', 'dvy', 'dvx', 'ca_corr', 'dist_c', 'nb_density_15', 'ca_corr_pcnt', 'cont', 'elong'), 
        variable_names=('dv (um/s)', 'dvz (um/s)', 'dvy (um/s)', 'dvx (um/s)', 'corrected calcium', 
                        'centre distance', 'density (platelets/um^2)', 'corrected calcium (%)', 'contraction (um/s)', 'elongation')
    ):
    if column is not None:
        df = df[df[column] == condition]
    for i, v in enumerate(variables):
        n = prefix + variable_names[i]
        out.loc[idx, n] = df[v].mean()
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



if __name__ == '__main__':
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    mips_n = '211206_mips_df_220818.parquet'
    dmso_n = '211206_veh-mips_df_220831.parquet'
    mpath = os.path.join(d, mips_n)
    dpath = os.path.join(d, dmso_n)
    #sp = os.path.join(d, 'short_mips.parquet') 
    paths = [mpath, dpath]
    #paths = [sp, ] # only 2 thrombi for debugging
    save_path = '/Users/amcg0011/Data/platelet-analysis/MIPS_surface/mips_dmso_positional_data-new.csv'
    positional_information_data(paths, save_path)
    #df = pd.read_csv(save_path)
    #txs = df['path'].apply(get_treatment_name)
    #df['treatment'] = txs
    #df.to_csv(save_path)
