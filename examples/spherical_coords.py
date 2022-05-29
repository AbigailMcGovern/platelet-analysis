from plateletanalysis.variables.transform import spherical_coordinates
from plateletanalysis.variables.measure import add_finite_diff_derivative
import pandas as pd
import os
from pathlib import Path

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
#sp = os.path.join(d, '211206_saline_df_nb.parquet')
#sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
#df = pd.read_parquet(sp)
#df = df.drop(columns=['phi', 'rho', 'theta'])
#df = spherical_coordinates(df)
#sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
#df.to_parquet(sp)
#df = add_finite_diff_derivative(df, 'rho')
#df = add_finite_diff_derivative(df, 'phi')
#df = add_finite_diff_derivative(df, 'theta')
#df.to_parquet(sp)


files = [ '211206_cang_df_spherical-coords.parquet', 
         '211206_biva_df_spherical-coords.parquet', '211206_ctrl_df_spherical-coords.parquet', 
         '211206_sq_df_spherical-coords.parquet', '211206_veh-sq_df_spherical-coords.parquet']
for f in files:
    print('adding spherical coordinates to ', f)
    sp = os.path.join(d, f)
    df = pd.read_parquet(sp)
    df = df.drop(columns=['rho_diff', 'theta_diff', 'phi_diff'])
    #df = spherical_coordinates(df)
    #n = Path(sp).stem + '_spherical-coords.parquet'
    #sp = os.path.join(d, n)
    #df.to_parquet(sp)
    print('adding spherical velocities to ', f)
    df = add_finite_diff_derivative(df, 'rho')
    df = add_finite_diff_derivative(df, 'phi')
    df = add_finite_diff_derivative(df, 'theta')
    df.to_parquet(sp)