import pandas as pd
import napari 
from tqdm import tqdm
from plateletanalysis.variables.transform import spherical_coordinates
from plateletanalysis.variables.measure import finite_difference_derivatives, add_finite_diff_derivative, contractile_motion
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density, local_calcium
from plateletanalysis.variables.transform import spherical_coordinates



def smooth_coords(df, window=10):
    its = 0
    for key, grp in df.groupby(['path', 'particle']):
        its += 1
    with tqdm(desc='Smoothing coordinates', total=its) as progress:
        for key, grp in df.groupby(['path', 'particle']):
            grp = grp.sort_values('frame')
            idx = grp.index.values
            new_x = grp['x_s'].rolling(window, win_type='triang',min_periods=1, center=False).mean()
            df.loc[idx, 'x_s_orig'] = grp['x_s'].values
            df.loc[idx, 'x_s'] = new_x
            new_y = grp['ys'].rolling(window, win_type='triang',min_periods=1, center=False).mean()
            df.loc[idx, 'ys_orig'] = grp['ys'].values
            df.loc[idx, 'ys'] = new_y
            new_z = grp['zs'].rolling(window, win_type='triang',min_periods=1, center=False).mean()
            df.loc[idx, 'zs_orig'] = grp['zs'].values
            df.loc[idx, 'zs'] = new_z
            progress.update(1)
    return df



def new_spherical_coords(df):
    if 'phi' in df.columns.values:
        df = df.drop(columns=['phi', 'rho', 'theta'])
    df = spherical_coordinates(df)
    return df



def new_velocities(df):
    if 'phi_diff' in df.columns.values:
        df = df.drop(columns=['phi_diff', 'rho_diff', 'theta_diff'])
    if 'dv' in df.columns.values:
        df = df.drop(columns=['dv', 'dvy', 'dvx', 'dvz'])
    if 'cont' in df.columns.values:
        df = df.drop(columns=['cont', 'cont_p'])
    df = finite_difference_derivatives(df)
    df = contractile_motion(df)
    df = add_finite_diff_derivative(df, 'phi')
    df = add_finite_diff_derivative(df, 'theta')
    df = add_finite_diff_derivative(df, 'rho')
    return df



def new_neighbours(df):
    if 'nb_particles_15' in df.columns.values:
        df = df.drop(columns=['nb_particles_15', 'nb_disp_15'])
    if 'nb_density_15' in df.columns.values:
        df = df.drop(columns=['nb_density_15', ])
    if 'nb_ca_corr_15' in df.columns.values:
        df = df.drop(columns=['nb_ca_corr_15', ])
    df = add_neighbour_lists(df, max_dist=15)
    df = add_neighbour_lists(df, max_dist=10)
    df = local_density(df, r=15)
    df = local_density(df, r=10)
    df = local_calcium(df, r=15)
    return df



# -------------
# VISUALISATION
# -------------


def get_tracks(df, cols=('frame', 'zs', 'ys', 'x_s')):
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



if __name__ == '__main__':
    import os
    d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
    mips_n = '211206_mips_df_220818.parquet'
    dmso_n = '211206_veh-mips_df_220831.parquet'
    mpath = os.path.join(d, mips_n)
    dpath = os.path.join(d, dmso_n)
    df = pd.read_parquet(dpath)
    #df = smooth_coords(df)
    save_path = os.path.join(d, 'dmso_df_smoothed-10.parquet')
    #df.to_parquet(save_path)

    #save_path = os.path.join(d, 'mips_df_smoothed-10.parquet')
    #save_path = os.path.join(d, 'mips_df_smoothed.parquet')
    #df.to_parquet(save_path)

    df = pd.read_parquet(save_path)
    if 'pid' not in df.columns.values:
        df['pid'] = range(len(df))
    df = new_spherical_coords(df)
    df.to_parquet(save_path)

    df = new_velocities(df)
    df.to_parquet(save_path)

    df = new_neighbours(df)
    df.to_parquet(save_path)

    df = df.dropna(subset=['dv', 'dvz', 'dvy', 'dvx', 'cont', 'phi_diff', 'rho_diff', 'theta_diff'])
    display_all_tracks(df)
