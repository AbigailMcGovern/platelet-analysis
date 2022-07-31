import matplotlib.pyplot as plt
import os
import napari
import pandas as pd
import numpy as np

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
path = os.path.join(d, '211206_saline_df_spherical-coords_density_pcnt.parquet')
df = pd.read_parquet(path)

def get_tracks(df, cols=('particle', 'frame', 'z_pixels', 'y_pixels', 'x_pixels')):
    tracks = df[list(cols)].values
    return tracks

def display_all_tracks(df):
    files = pd.unique(df['path'])
    v = napari.Viewer()
    for f in files:
        f_df = df[df['path'] == f]
        tracks = get_tracks(f_df, ('particle', 'frame', 'zs', 'ys', 'x_s'))
        v.add_tracks(tracks, properties=f_df, name=f, visible=False)

df = df[df['nrtracks'] > 4]
df = df[(df['phi_diff'] > -0.0825) & (df['phi_diff'] < 0.0825)]
df = df[(df['theta_diff'] > -0.0407) & (df['theta_diff'] < 0.0407)]
df = df.dropna(subset=['phi_diff', 'rho_diff', 'theta_diff'])

display_all_tracks(df)
napari.run()