import pandas as pd
import os
from pathlib import Path
import re
import numpy as np



df_dict={
    'pid':'pid',
    'path':'file',
    'frame':'t',
    'x_s':'ys',
    'ys':'xs',
    'zs':'zs',
    'particle':'particle',
    'nrtracks':'track_no_frames',
    'c0_mean':'GaAsP Alexa 488: mean_intensity',
    'c0_max':'GaAsP Alexa 488: max_intensity',
    'c1_mean':'GaAsP Alexa 568: mean_intensity',
    'c1_max':'GaAsP Alexa 568: max_intensity',
    'c2_mean':'Alxa 647: mean_intensity',
    'c2_max':'Alxa 647: max_intensity',
    'vol':'volume',
    'elong':'elongation',
    'flatness':'flatness',
    'treatment':'treatment',
    'cohort':'cohort',
    'eigval_0':'inertia_tensor_eigvals-0',
    'eigval_1':'inertia_tensor_eigvals-1',
    'eigval_2':'inertia_tensor_eigvals-2',
    }



def get_experiment_df(
        directory, 
        treatment, 
        meta_suffix='md.csv', 
        tracks_suffix='tracks.csv', 
        pid_col='pid',
        tx_col='treatment', 
        file_col='file',
        ):
    '''
    Get a dataframe for a treatment or list of treatments of choice
    '''
    meta_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(directory)
        for name in files
        if name.endswith(meta_suffix)
    ]
    tracks_files = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(directory)
        for name in files
        if name.endswith(tracks_suffix)
    ]
    meta_df = [pd.read_csv(p) for p in meta_files]
    meta_df = pd.concat(meta_df, ignore_index=True, names=pid_col).reset_index(drop=True)
    if isinstance(treatment, str):
        treatment = [treatment, ]
    filelist = [[file for file in meta_df[meta_df[tx_col]==t][file_col].unique()] for 
            t in meta_df[tx_col].unique() if t in treatment]
    filelist = concatenate(filelist)
    pathlist = [path for path in tracks_files for f in filelist if f in path]
    df = [pd.read_csv(p) for p in pathlist]
    df = pd.concat(df, ignore_index=True, names=pid_col).reset_index(drop=True)
    df_new=pd.DataFrame()
    for new , old in df_dict.items():
        df_new[new] = df[old]
    del df
    df_new = df_new.drop(pid_col, axis=1).reset_index().rename({'index' : pid_col}, axis=1)
    return df_new, meta_df


def add_info_from_file_name(
    df,
    var=('date', 'mouse', 'inj', 'inh', 'exp'), 
    positions = (0, 1, 2, 3, 4),
    types=(str, np.int64, np.int64, str, str),
    dec_to_str = (False, False, False, False, True),
    path_col='path'
    ):
    for v in var:
        df['v'] = [None, ] * len(df)
    for p in pd.unique(df[path_col]):
        result = split_and_find(p, positions, var, types, dec_to_str)
        idxs = df[df[path_col] == p].index.values
        for v in result.keys():
            df.loc[idxs, v] = result[v]
    for v, t in zip(var, types):
        df[v] = df[v].astype(t)
    return df

# C2aKD --> date mouse inh inj exp (0, 1, 3, 2, 4)
# 201125_IVMTR83_Inj11_CMFDA_DMSO_exp3.nd2

def split_and_find(s, idxs, var, types, dec_2_s):
    l = s.split('_')
    res = {}
    digits = re.compile(r'\d+')
    for i, v, t, d2s in zip(idxs, var, types, dec_2_s):
        part = l[i]
        match = digits.findall(part)
        get_dec = t in (np.int64, np.int32, int, np.float64) or d2s
        if get_dec and len(match) > 0:
            info = match[0]
            if not d2s:
                info = int(info)
        else:
            info = part
        res[v] = info
    return res


def fix_names(s: str):
    new = s.replace(' ', '')
    new = new.replace('-', '')
    return new

# df['path'] = df['path'].apply(fix_names)


# -------
# Helpers
# -------

def concatenate(l):
    r = []
    for i in l:
        r = r + i
    return r

