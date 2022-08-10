import pandas as pd
import argparse
from plateletanalysis.variables.measure import finite_difference_derivatives, add_finite_diff_derivative, \
    contractile_motion, path_disp_n_tortuosity


p = argparse.ArgumentParser()
p.add_argument('-f', '--file', help='parquet file platelet info')
p.add_argument('-s', '--save', help='parquet file into which to save output')
args = p.parse_args()

p = args.file
sp = args.save
df = pd.read_parquet(p)

df = finite_difference_derivatives(df)
df = contractile_motion(df)
df = path_disp_n_tortuosity(df)

df = add_finite_diff_derivative(df, 'phi')
df = add_finite_diff_derivative(df, 'theta')
df = add_finite_diff_derivative(df, 'rho')
df = add_finite_diff_derivative(df, 'ca_corr')

df.to_parquet(sp)