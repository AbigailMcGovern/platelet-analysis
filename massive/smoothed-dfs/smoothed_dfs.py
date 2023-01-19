import pandas as pd
import napari 
from tqdm import tqdm
from plateletanalysis.variables.transform import spherical_coordinates
from plateletanalysis.variables.measure import finite_difference_derivatives, add_finite_diff_derivative, contractile_motion, \
     quantile_normalise_variables_frame, quantile_normalise_variables
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


def quantile_normalisations(df):
    df = quantile_normalise_variables(df, ['x_s', 'ys', 'zs'])
    df = quantile_normalise_variables_frame(df, ['nb_density_15', 'nb_density_10'])
    return df


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--file', help='parquet file platelet info')
    p.add_argument('-s', '--save', help='parquet file into which to save output')
    args = p.parse_args()
    df_path = args.file
    save_path = args.save
    df = pd.read_parquet(df_path)
    # smooth coords
    df = smooth_coords(df)
    df.to_parquet(save_path)
    print(f'Computed smoothed coords for {df_path}')
    # add spherical coords
    df = new_spherical_coords(df)
    df.to_parquet(save_path)
    print(f'Computed spherical coords for {df_path}')
    # recacluate velocities
    df = new_velocities(df)
    df.to_parquet(save_path)
    print(f'Computed velocities for {df_path}')
    # neighbour variables
    df = new_neighbours(df)
    df.to_parquet(save_path)
    print(f'Computed neighbours for {df_path}')
    # quantile normalisation
    df = quantile_normalisations(df)
    df.to_parquet(save_path)
    print(f'Computed quantile normalisation for {df_path}')
