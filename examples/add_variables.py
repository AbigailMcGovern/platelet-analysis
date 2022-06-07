from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from plateletanalysis.variables.measure import path_disp_n_tortuosity
import os
import pandas as pd


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(sp)
#df = local_density(df)
#df.to_parquet(sp)

#df['nrterm'] = df['nrtracks'] - df['tracknr']
#df.to_parquet(sp)

df = path_disp_n_tortuosity(df)
df.to_parquet(sp)