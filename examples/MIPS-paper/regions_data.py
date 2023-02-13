import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolz import curry
from plateletanalysis.variables.basic import quantile_normalise_variables
from plateletanalysis.variables.neighbours import local_density, add_neighbour_lists
from plateletanalysis.variables.transform import spherical_coordinates
import os
from pathlib import Path
from tqdm import tqdm
from scipy.stats import scoreatpercentile


# Values to get 

def count(grp):
    return len(grp)


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
    val = len(df[df['tracknr'] == 1]) / 0.321764322705706
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


def sliding(df):
    return df['sliding (ums^-1)'].mean()


def p_lt15s(df):
    lt15 = len(df[df['total time tracked (s)'] < 15])
    t = len(df)
    #gt30 = len(df[df['time (s)'] >= 30])
    return lt15 / t


def p_gt60s(df):
    gt60 = len(df[df['total time tracked (s)'] > 60])
    t = len(df)
    #gt30 = len(df[df['time (s)'] >= 30])
    return gt60 / t


def tracking_time_IQR(df):
    Q1 = scoreatpercentile(df['tracking time (s)'].values, 25)
    Q3 = scoreatpercentile(df['tracking time (s)'].values, 75)
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
    return p / t


def p_recruited_gt60(df):
    p = len(df[(df['total time tracked (s)'] > 60) & (df['tracknr'] == 1)])
    t = len(df[df['tracknr'] == 1])
    return p / t


# looping function

def loop_over_exp_region(df, exp_col, time_col, val_col, val_func, other_cols):
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
    with tqdm(total=nits) as progress:
        for k, g in df.groupby([exp_col, time_col, 'region']): # would have done groupby apply, but I don't care
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


def generate_var_dict():
    var_dict = {
        #'platelet count' : [count, None], 
        #'platelet density (um^-3)' : [density, ['nb_density_15', ]], 
        #'thrombus edge distance (um)' : [outer_edge, ['rho', (90, 98)]], 
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
    return var_dict


# get all data

def lineplots_regions_data_all(
        df, 
        save_path,
        var_dict, 
        treatements=('MIPS', 'SQ', 'cangrelor'),
        controls=('DMSO (MIPS)', 'DMSO (SQ)', 'saline'),
        exp_col='path',
        time_col='time (s)',
        track_lim=10,
        other_cols=('treatment', ), 
        #names=('platelet count', 'platelet density (um^-3)', 'thrombus edge distance (um)', 
         #      'recruitment (s^-1)', 'shedding (s^-1)', 'mean stability', 
          #     'dvy (um s^-1)', 'mean tracking time (s)', 'sliding (ums^-1)', 
          #      '
          # ), 
        ):
    '''
    other_cols: tuple
        other columns to collect values for. Takes only the first value in group. 
        The first value in other_cols will be used as the hue for sns.lineplot. 
    '''
    var_dict = generate_var_dict()
    names = list(var_dict.keys())
    funcs = [var_dict[n][0] for n in names]
    curry_with = [var_dict[n][1] for n in names]
    df = df[df['nrtracks'] > track_lim]
    df = add_region_category(df)
    data = []
    ind_save = [os.path.join(Path(save_path).parents[0], Path(save_path).stem + f'_{n}.csv') for n in names]
    for i, func in enumerate(funcs):
        if not os.path.exists(ind_save[i]):
            n = names[i]
            print(f'Getting values for {n}...')
            cw = curry_with[i]
            if cw is not None:
                func = func(*cw)
            result = loop_over_exp_region(df, exp_col,time_col, n, func, other_cols)
            result.to_csv(ind_save[i])
        else:
            result = pd.read_csv(ind_save[i])
        result = result.set_index([exp_col, time_col, 'region'], drop=True)
        data.append(result)
        #TODO: separate script for plotting data
    data = pd.concat(data, axis=1)
    data = data.reset_index(drop=False)
    tx = data[other_cols[0]].values
    if 'Unnamed: 0' in data.columns.values:
        data = data.drop(columns=['Unnamed: 0', ])
    data = data.drop(columns=[other_cols[0], ])
    data[other_cols[0]] = tx[:, 0]
    data[:10].to_csv('/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/debugging.csv')
    # percentage data
    for n in names:
        nn = f'{n} pcnt'
        data[nn] = [None, ] * len(data)
    nits = len(treatements) * len(pd.unique(data[time_col])) * len(pd.unique(data['region'])) * len(names)
    print('Getting percent of vehicle scores...')
    with tqdm(total=nits) as progress:
        for i, tx in enumerate(treatements):
            v = controls[i]
            tx_df = data[data[other_cols[0]] == tx]
            for t, g in tx_df.groupby([time_col, 'region']):
                idxs = g.index.values
                v_df = data[(data[other_cols[0]] == v) & (data[time_col] == t[0]) & (data['region'] == t[1])]
                v_means = {n : v_df[n].mean() for n in names}
                for n in names:
                    nn = f'{n} pcnt'
                    pcnt = g[n].values / v_means[n] * 100
                    data.loc[idxs, nn] = pcnt
                    progress.update(1)
    data.to_csv(save_path)


# add variables


def add_spherical_and_local_dens(data_dir, file_names):
    file_paths = [os.path.join(data_dir, n) for n in file_names]
    dfs = [pd.read_parquet(p) for p in file_paths]
    for p, df in zip(file_paths, dfs):
        if 'pid' not in df.columns.values:
            df['pid'] = range(len(df))
        if 'rho' not in df.columns.values:
           print(f'Getting spherical coords for {p}')
           df = spherical_coordinates(df)
           df.to_csv(p)
        if 'nb_density_15' not in df.columns.values:
           print(f'Getting neighbours for {p}')
           df = add_neighbour_lists(df)
           print(f'Getting density for {p}')
           df = local_density(df)
           df.to_csv(p)



def add_region_category(df):
    rcyl = (df.x_s ** 2 + df.ys ** 2) ** 0.5
    df['rcyl'] = rcyl
    df['region'] = [None, ] * len(df)
    # center
    sdf = df[df['rcyl'] <= 37.5]
    idxs = sdf.index.values
    df.loc[idxs, 'region'] = 'center'
    # outer regions
    # 45 degrees = 0.785398
    sdf = df[df['rcyl'] > 37.5]
    # anterior
    rdf = sdf[sdf['phi'] > 0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'anterior'
    # lateral
    rdf = sdf[(sdf['phi'] < 0.785398) & (sdf['phi'] > -0.785398)]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'lateral'
    # posterior
    rdf = sdf[sdf['phi'] < -0.785398]
    idxs = rdf.index.values
    df.loc[idxs, 'region'] = 'posterior'
    return df


def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df


def add_sliding_variable(df):
    df['sliding (ums^-1)'] = [None, ] * len(df)
    # not moving in direction of blood flow
    sdf = df[df['dvy'] >= 0]
    idxs = sdf.index.values
    df.loc[idxs, 'sliding (ums^-1)'] = 0
    # moving in the direction of blood floe
    sdf = df[df['dvy'] < 0]
    idxs = sdf.index.values
    new = np.abs(sdf['dvy'].values)
    df.loc[idxs, 'sliding (ums^-1)'] = new
    return df


def tracking_time_var(df):
    df['tracking time (s)'] = df['tracknr'] / 0.321764322705706
    return df


def time_tracked_var(df):
    df['total time tracked (s)'] = df['nrtracks'] / 0.321764322705706
    return df


def add_terminating(df):
    df['terminating'] = df['nrtracks'] == df['tracknr']
    return df



if __name__ == '__main__':
    from plateletanalysis.variables.basic import get_treatment_name, time_minutes

    # ------------------
    # Get data from file
    # ------------------
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
                  '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
                  '211206_sq_df.parquet', '211206_veh-sq_df.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    #dfs = [pd.read_parquet(p) for p in file_paths]
    data = []
    for p in file_paths:
        df = pd.read_parquet(p)
        data.append(df)
    df = pd.concat(data).reset_index(drop=True)
    del data


    # -------------
    # Add variables
    # -------------
    df['treatment'] = df['path'].apply(get_treatment_name)
    df = df[df['treatment'] != 'DMSO (salgav)']
    df = add_time_seconds(df)
    df = add_terminating(df)
    df = add_sliding_variable(df)
    df = time_minutes(df)
    df = time_tracked_var(df)
    df = tracking_time_var(df)

    # ------------------
    # Compute data frame
    # ------------------
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230211_regionsdata_8var.csv'
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var_trk1.csv'
    #lineplots_regions_data_all(df, save_path, track_lim=1)
    #save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_8var_trk1_minute.csv'
    save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/regions_data/230212_regionsdata_12var_trk1_minute.csv'
    var_dict = generate_var_dict()
    lineplots_regions_data_all(df, save_path, var_dict, track_lim=1, time_col='minute')