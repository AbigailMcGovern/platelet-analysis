import matplotlib.pyplot as plt
import os
import napari
import pandas as pd
import numpy as np

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
mips_n = '211206_mips_df_220818.parquet'
saline_n = '211206_saline_df_220827_amp0.parquet'
path = os.path.join(d, saline_n)
df = pd.read_parquet(path)

def get_tracks(df, cols):
    tracks = df[list(cols)].values
    return tracks


def display_all_tracks_with_surface(df, var='nb_density_15_pcntf', thresh=50):
    files = pd.unique(df['path'])
    v = napari.Viewer()
    for f in files:
        f_df = df[df['path'] == f]
        tracks = get_tracks(f_df, ('particle', 'frame', 'zs', 'ys', 'x_s'))
        v.add_tracks(tracks, properties=f_df, name=f, visible=False)
        display_surface_points(f_df, f, v, var=var, thresh=thresh)
        display_inner_points(f_df, f, v, var=var, thresh=thresh)
    napari.run()


def display_surface_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[fdf[var] < thresh]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='white')


def display_inner_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[fdf[var] > thresh]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='red')




df = df[df['nrtracks'] > 10]
#df = df[(df['phi_diff'] > -0.0825) & (df['phi_diff'] < 0.0825)]
#df = df[(df['theta_diff'] > -0.0407) & (df['theta_diff'] < 0.0407)]
#df = df[(df['nb_cont_15'] > -1.8032318892514754) & (df['nb_cont_15'] < 3.6064637785029507)]
#df = df.dropna(subset=['phi_diff', 'rho_diff', 'theta_diff', 'nb_cont_15'])

display_all_tracks_with_surface(df)

#df1 = df1[df1['nrtracks'] > 4]
#df1 = df1[(df1['phi_diff'] > -0.0825) & (df1['phi_diff'] < 0.0825)]
#df1 = df1[(df1['theta_diff'] > -0.0407) & (df1['theta_diff'] < 0.0407)]
#df1 = df1.dropna(subset=['phi_diff', 'rho_diff', 'theta_diff'])