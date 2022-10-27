from numpy import var
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
import pandas as pd
import os
from pathlib import Path


def add_local_dens_and_pcnt_for_PH(df, sp):
    df = add_neighbour_lists(df)
    df = local_density(df)
    df.to_parquet(sp)
    df = quantile_normalise_variables(df, vars=('x_s', 'ys' ))
    df = quantile_normalise_variables_frame(df, vars=('nb_density_15', ))
    df.to_parquet(sp)

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'

#dataframes = ['211206_ctrl_df.parquet', '211206_cang_df.parquet', '211206_sq_df.parquet']
dataframes = ['211206_ctrl_df.parquet', '211206_par4--_df.parquet', 
              '211206_par4--biva_df.parquet', '211206_salgav_df.parquet', 
              '211206_salgav-veh_df.parquet', '211206_veh-sq_df.parquet']

for sn in dataframes:
    sp = os.path.join(d, sn)
    df = pd.read_parquet(sp)
    add_local_dens_and_pcnt_for_PH(df, sp)