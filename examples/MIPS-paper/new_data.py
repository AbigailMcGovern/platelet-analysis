import pandas as pd
import os
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import napari 


n = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/230301_MIPS_and_DMSO.parquet'
m = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_mips_df.parquet'
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_veh-mips_df.parquet'

def add_time_seconds(df, frame_col='frame'):
    df['time (s)'] = df[frame_col] / 0.321764322705706
    return df

new_df = pd.read_parquet(n)
new_df['group'] = 'new'

old_mips = pd.read_parquet(m)
old_mips['group'] = 'old'

old_dmso = pd.read_parquet(d)
old_dmso['group'] = 'old'

all_dfs = pd.concat([new_df, old_dmso, old_mips]).reset_index(drop=True)

def timeplot_counts(df):
    counts = {
        'path' : [], 
        'time (s)' : [], 
        'treatment' : [], 
        'group' : [],
        'count' : []
    }
    gb = ['path', 'time (s)', 'treatment', 'group']
    for k, grp in df.groupby(gb):
        for i, col in enumerate(gb):
            counts[col].append(k[i])
        c = len(pd.unique(grp['particle']))
        counts['count'].append(c)
    counts = pd.DataFrame(counts)
    fig, axs = plt.subplots(2, 4)
    mips = counts[counts['treatment'] == 'MIPS']
    dmso = counts[counts['treatment'] == 'DMSO (MIPS)']
    dfs = [mips, dmso]
    for i, grp in enumerate(dfs):
        
        sns.lineplot(data=grp, x='time (s)')


paths_old = pd.unique(old_dmso['path'])
paths_new = pd.unique(new_df['path'])
#new_df['ys'] = - new_df['ys']
old_tracks = old_dmso[old_dmso['path'] == paths_old[0]][['particle', 'frame', 'zs', 'ys', 'x_s']].values
new_tracks = new_df[new_df['path'] == paths_new[0]][['particle', 'frame', 'zs', 'ys', 'x_s']].values


V = napari.view_tracks(old_tracks)
V.add_tracks(new_tracks)
napari.run()

