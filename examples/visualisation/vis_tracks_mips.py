import matplotlib.pyplot as plt
import os
import napari
import pandas as pd
import numpy as np

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
saline_n = '211206_saline_df_220614-amp0.parquet'
saline_s5 = '211206_saline_df_smooth-5_0.parquet'
biva_n = '220603_211206_biva_df_spherical-coords.parquet'
cang_n = '220603_211206_cang_df_spherical-coords.parquet'
mips_n = '211206_mips_df_220818.parquet'
path = os.path.join(d, mips_n)
df = pd.read_parquet(path)

def get_tracks(df, cols):
    tracks = df[list(cols)].values
    return tracks

def display_all_tracks(df):
    files = pd.unique(df['path'])
    v = napari.Viewer()
    for f in files:
        f_df = df[df['path'] == f]
        tracks = get_tracks(f_df, ('particle', 'frame', 'zs', 'ys', 'x_s'))
        v.add_tracks(tracks, properties=f_df, name=f, visible=False)
    napari.run()

df = df[df['nrtracks'] > 10]
#df = df[(df['phi_diff'] > -0.0825) & (df['phi_diff'] < 0.0825)]
#df = df[(df['theta_diff'] > -0.0407) & (df['theta_diff'] < 0.0407)]
#df = df[(df['nb_cont_15'] > -1.8032318892514754) & (df['nb_cont_15'] < 3.6064637785029507)]
#df = df.dropna(subset=['phi_diff', 'rho_diff', 'theta_diff', 'nb_cont_15'])

display_all_tracks(df)

#df1 = df1[df1['nrtracks'] > 4]
#df1 = df1[(df1['phi_diff'] > -0.0825) & (df1['phi_diff'] < 0.0825)]
#df1 = df1[(df1['theta_diff'] > -0.0407) & (df1['theta_diff'] < 0.0407)]
#df1 = df1.dropna(subset=['phi_diff', 'rho_diff', 'theta_diff'])