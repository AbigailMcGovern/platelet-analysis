from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from plateletanalysis.variables.measure import quantile_normalise_variables, quantile_normalise_variables_frame
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
#sp = os.path.join(d, '211206_mips_df.parquet')
vn = '211206_veh-mips_df.parquet'
sp = os.path.join(d, vn)
df = pd.read_parquet(sp)


#sn = '211206_mips_df_220818.parquet'
sn = '211206_veh-mips_df_220831.parquet'
sp = os.path.join(d, sn)
df = add_neighbour_lists(df)
df = local_density(df)
df.to_parquet(sp)
df = quantile_normalise_variables(df, vars=('nb_density_15', ))
df.to_parquet(sp)