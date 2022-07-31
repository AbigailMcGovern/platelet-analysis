from plateletanalysis.variables.measure import quantile_normalise_variables
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
n = '211206_saline_df_220611-0.parquet'
p = os.path.join(d, n)
df = pd.read_parquet(p)


sn = '211206_saline_df_220612-amp0.parquet'
sp = os.path.join(d, sn)
df = quantile_normalise_variables(df, vars=('phi', 'phi_diff', 'rho', 'rho_diff', 'theta', 'theta_diff', 'zs', 'fibrin_dist', 'fibrin_cont'))
df.to_parquet(sp)

# df = quantile_normalise_variables(df, vars=('emb_prox_k10', 'nb_ca_copying_15'))