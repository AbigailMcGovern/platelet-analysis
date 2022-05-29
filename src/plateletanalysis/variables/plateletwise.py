import pandas as pd
import numpy as np





def construct_platelet_df(
    df, 
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
        'mean_dx', 
        'mean_dy', 
        'mean_dz', 
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
        'var_dx', 
        'var_dy', 
        'var_dz', 
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
        'end_tort'
    ), 
    save=None):
    '''
    Highly inefficient function to generate platelet DF. Will make more efficient only if
    too time intensive when run. 
    '''
    df = df[df['tracked'] == True] # only interested in tracked platelets
    df_gb = df.groupby(['path', 'particle'])
    idx = pd.unique(df_gb.index.values)
    plate_id = range(len(idx))
    pdf = {
        'plate_id' : plate_id, 
        'path' : [i[0] for i in idx], 
        'particle' : [i[1] for i in idx], 
        'gbindex' : idx
        }
    pdf = pd.DataFrame(pdf)
    pdf.set_index('gbindex')
    for col in cols:
        first_val = df_gb.loc[idx[0], col]
        if col.startswith('sum_'):
            bcol = col[4:]
            idxs, vals = cumulative_platelet_score(df_gb, col)
            pdf.loc[idxs, col] = vals
        elif col.startswith('start_'):
            bcol = col[6:]
            idxs, vals = start_track_value(df_gb, bcol)
            pdf.loc[idxs, col] = vals
        elif col.startswith('end_'):
            bcol = col[4:]
            idxs, vals = end_track_value(df_gb, bcol)
            pdf.loc[idxs, col] = vals
        elif col.startswith('var_'):
            bcol = col[4:]
            idxs, vals = track_varience(df_gb, bcol)
            pdf.loc[idxs, col] = vals
        elif col.startswith('mean_'):
            bcol = col[5:]
            means = df_gb.mean()
            idxs = means.index.values
            pdf.loc[idxs, col] = means
        else:
            idxs, vals = catagorical_vars(df_gb, col)
            pdf.loc[idxs, col] = vals
    pdf = pdf.reset_index()
    if save is not None:
        pdf.to_parquet(save)
    return pdf



def cumulative_platelet_score(df_gb, col):
    #df_gb = df.groupby(['path', 'particle'])
    idx = pd.unique(df_gb.index.values)
    vals = df_gb[col].sum()
    return idx, vals


def start_track_value(df_gb, col):
    #df_gb = df.groupby(['path', 'particle'])
    df_gb = df_gb[df_gb['tracknr'] == 1] # each platelet should only have one start track
    idx = pd.unique(df_gb.index.values)
    vals = df[col].values
    return idx, vals


def end_track_value(df_gb, col):
    #df_gb = df.groupby(['path', 'particle'])
    df_gb = df_gb[df_gb['terminating'] == True] # each platelet should only have one terminating track
    idx = pd.unique(df_gb.index.values)
    vals = df[col].values
    return idx, vals


def track_varience(df_gb, col):
    #df_gb = df.groupby(['path', 'particle'])
    idx = pd.unique(df_gb.index.values)
    sem = df_gb[col].sem
    vals = sem ** 2
    return idx, vals


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


