from plateletanalysis.variables.transform import spherical_coordinates
from plateletanalysis.variables.measure import add_finite_diff_derivative
import pandas as pd
import os

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_nb.parquet')
df = pd.read_parquet(sp)
df = spherical_coordinates(df)
sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df.to_parquet(sp)
df = add_finite_diff_derivative(df, 'rho')
df = add_finite_diff_derivative(df, 'phi')
df = add_finite_diff_derivative(df, 'theta')
df.to_parquet(sp)