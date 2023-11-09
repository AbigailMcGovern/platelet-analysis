import pandas as pd
import os
import numpy as np
import napari



def smooth_vars(df, vars, w=15, t='time (s)', gb=['path', 'particle'], add_suff=None):
    df = df.sort_values(t)
    for v in vars:
        if add_suff is not None:
            v_n = v + add_suff
        else:
            v_n = v
        for k, grp in df.groupby(gb):
            rolled = grp[v].rolling(window=w, center=True).mean()
            idxs = grp.index.values
            df.loc[idxs, v_n] = rolled
    return df


def local_densification(df):
    df = df.sort_values('time (s)')
    for k, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        vals = np.diff(grp['nb_density_15'].values)
        vals = np.concatenate([[np.nan, ], vals])
        vals = vals * 0.32
        df.loc[idx, 'densification (/um3/sec)'] = vals
    df = smooth_vars(df, ['densification (/um3/sec)', ], 
                     w=15, t='time (s)', gb=['path', 'particle'], 
                     add_suff=None)
    return df


# read data
sp_new = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/p-sel_cleaned_with_new.parquet'
df = pd.read_parquet(sp_new)

# new vars
df = local_densification(df)
df = smooth_vars(df, ['densification (/um3/sec)', ])
df = df.fillna(0)

# variables for view
track_dfs = []
props = []
paths = []
for k, grp in df.groupby('path'):
    paths.append(k)
    track_dfs.append(grp[['particle', 'frame', 'zs', 'ys', 'x_s']].values)
    props.append(grp[['nb_cont_15', 'av_nb_disp_15', 'nb_density_15', 'p-sel average intensity', 'densification (/um3/sec)']])

# view
v = napari.Viewer()
for tdf, pdf, p in zip(track_dfs, props, paths):
    pdf = pdf.to_dict(orient='list')
    v.add_tracks(data=tdf, name=p, properties=pdf, visible=False)
napari.run()