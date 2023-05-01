import pandas as pd
import numpy as np
from tqdm import tqdm
import os




def construct_platelet_df(
    df, 
    save,
    cols=(
        'mean_phi', 
        'mean_theta', 
        'mean_rho', 
        'mean_phi_diff', 
        'mean_theta_diff', 
        'mean_rho_diff', 
        'mean_x_s', 
        'mean_ys', 
        'mean_zs', 
        'mean_dv', 
        'mean_dvx', 
        'mean_dvy', 
        'mean_dvz', 
        'mean_ca_corr', 
        'mean_elong', 
        'mean_flatness', 
        'mean_c1_mean', 
        'mean_c2_mean', 
        'var_phi', 
        'var_theta', 
        'var_rho', 
        'var_phi_diff', 
        'var_theta_diff', 
        'var_rho_diff', 
        'var_x_s', 
        'var_ys', 
        'var_zs', 
        'var_dv', 
        'var_dvx', 
        'var_dvy', 
        'var_dvz', 
        'var_ca_corr', 
        'start_frame', 
        'start_rho', 
        'start_phi', 
        'start_theta', 
        'end_rho', 
        'end_phi', 
        'end_theta', 
        'end_path_len', 
        'end_disp', 
        'end_tort', 
        'end_frame', 
        'mean_nb_density_15',
        'var_nb_density_15',
        'mean_n_neighbours', 
        'var_n_neighbours', 
        'start_nrtracks', 
        'mean_stab', 
        'var_stab', 
    ), 
    ):
    '''
    Highly inefficient function to generate platelet DF. Will make more efficient only if
    too time intensive when run. 
    '''
    df = df[df['tracked'] == True] # only interested in tracked platelets
    #df_gb = df.groupby(['path', 'particle'])
    df_gb = df.set_index(['path', 'particle']).sort_index()
    idx = pd.unique(df_gb.index.values)
    if not os.path.exists(save):
        plate_id = range(len(idx))
        pdf = {
            'plate_id' : plate_id, 
            'path' : [i[0] for i in idx], 
            'particle' : [i[1] for i in idx], 
            }
        pdf = pd.DataFrame(pdf)
    else:
        pdf = pd.read_parquet(save)
    pdf = pdf.set_index(['path', 'particle'])
    cols = [c for c in cols if c not in pdf.columns.values]
    n_iter = len(idx) * len(cols)
    with tqdm(total=n_iter) as progress:
        for i in idx:
            idf = df_gb.loc[i, :]
            for col in cols:
                if col.startswith('sum_'):
                    bcol = col[4:]
                    val = idf[bcol].sum()
                    pdf.loc[i, col] = val
                    progress.update(1)
                elif col.startswith('start_'):
                    bcol = col[6:]
                    val = start_track_value(idf, bcol, i)
                    pdf.loc[i, col] = val
                    progress.update(1)
                elif col.startswith('end_'):
                    bcol = col[4:]
                    val = end_track_value(idf, bcol, i)
                    pdf.loc[i, col] = val
                    progress.update(1)
                elif col.startswith('var_'):
                    bcol = col[4:]
                    val = track_varience(idf, bcol, i)
                    pdf.loc[i, col] = val
                    progress.update(1)
                elif col.startswith('mean_'):
                    bcol = col[5:]
                    mean = idf[bcol].mean()
                    pdf.loc[i, col] = mean
                    progress.update(1)
                elif col.startswith('std_'):
                    bcol = col[4:]
                    std = idf[bcol].std()
                    pdf.loc[i, col] = std
                    progress.update(1)
                #else:
                 #   idxs, vals = catagorical_vars(df_gb, col)
                  #  pdf.loc[idxs, col] = vals
                   # progress.update(1)
        if save is not None:
            cpdf = pdf.reset_index()
            cpdf.to_parquet(save)
    pdf = pdf.reset_index()
    if save is not None:
        pdf.to_parquet(save)
    return pdf



def start_track_value(idf, col, idx):
    #df_gb = df.groupby(['path', 'particle'])
    df = idf[idf['tracknr'] == 1] # each platelet should only have one start track
    try:
        assert len(df) == 1
        val = df[col].sum()
    except:
        print(f'{idx} has no start track... adding NaN')
        val = np.NaN
    return val


def end_track_value(idf, col, idx):
    #df_gb = df.groupby(['path', 'particle'])
    df = idf[idf['terminating'] == True] # each platelet should only have one terminating track
    try:
        assert len(df) == 1
        val = df[col].sum()
    except:
        print(f'{idx} has no start track... adding NaN')
        val = np.NaN
    return val


def track_varience(idf, col, idx):
    #df_gb = df.groupby(['path', 'particle'])
    sem = idf[col].sem()
    sem = np.array(sem)
    val = sem ** 2
    return val


def catagorical_vars(df_gb, col): 
    idx = pd.unique(df_gb.index.values)
    vals = []
    for i in idx:
        sdf = df_gb.loc[i]
        u = pd.unique(sdf[col])
        if len(u) == 1:
            u = u[0]
        vals.append(u)
    return idx, vals


# ------------
# Rolling mean
# ------------

def rolling_mean(df, col, w=3):
    paths = pd.unique(df['path'])
    for p in paths:
        pdf = df[df['path'] == p]
        particles = pd.unique(pdf['particle'])
        for pa in particles:
            padf = pdf[pdf['particle'] == pa]
            padf = padf.sort_values(by='frame')
            idxs = padf.index.values
            df.loc[idxs, col]