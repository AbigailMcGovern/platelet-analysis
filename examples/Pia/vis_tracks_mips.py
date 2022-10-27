
import matplotlib.pyplot as plt
import os
import napari
import pandas as pd
import numpy as np
from plateletanalysis.variables.measure import quantile_normalise_variables_frame
from plateletanalysis.variables.transform import spherical_coordinates

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
mips_n = '211206_mips_df_220818.parquet'
saline_n = '211206_saline_df_220827_amp0.parquet'
biva_n = '220603_211206_biva_df_spherical-coords.parquet'
cang_n = '220603_211206_cang_df_spherical-coords.parquet'
sq_n = '211206_sq_df.parquet'


path = os.path.join(d, mips_n)
df = pd.read_parquet(path)
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
    return df

def add_vars(df, thresh=50, var='nb_desnity_15_pcntf'):
    files = pd.unique(df['path'])
    for f in files:
        fdf = df[df['path'] == f]
        sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40)]


def get_tracks(df, cols=('frame', 'zs', 'ys', 'x_s')):
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
        display_donut_points(f_df, f, v, var=var, thresh=thresh)
        display_anterior_surface_points(f_df, f, v, var=var, thresh=thresh)
        display_tail_points(f_df, f, v, var=var, thresh=thresh)
    napari.run()


def display_surface_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40)]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='white')


def display_inner_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[(fdf[var] > thresh)]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='red')


def display_donut_points(fdf, f, v, var='nb_density_15_pcnt', thresh=40):
    # MIPS = 20-29
    # SQ = 15-26
    # biva = 5-15
    # cang = 4-14
    # saline = 
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] < 30) & (fdf['zs'] < 56) & (fdf['frame'] > 24) & (fdf['frame'] < 32)]
    particles = sdf['particle'].values
    fdf = fdf.set_index('particle')
    fdf['donut_p'] = False
    fdf.loc[particles, 'donut_p'] = True
    fdf = fdf.reset_index()
    sdf = fdf[fdf['donut_p'] == True]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='green')


def display_anterior_surface_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40) & (fdf['ys'] > 0)]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='blue')



def display_tail_points(fdf, f, v, var='nb_density_15_pcnt', thresh=50):
    sdf = fdf[(fdf[var] < thresh) & (fdf['zs'] > 8) & (fdf['dist_c_pcntf'] > 40) & (fdf['phi'] < -0.78539)]
    points = get_tracks(sdf, ('frame', 'zs', 'ys', 'x_s'))
    v.add_points(points, properties=sdf, name=f, visible=False, size=2, edge_color='pink')


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