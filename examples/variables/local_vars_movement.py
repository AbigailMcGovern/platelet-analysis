from plateletanalysis.variables.neighbours import local_variable_mean
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220818_amp0.parquet'
#df = pd.read_parquet(sp)

sp = '/Users/amcg0011/Data/platelet-analysis/dataframes/211206_saline_df_220827_amp0.parquet'
df = pd.read_parquet(sp)

#df = local_variable_mean(df, 'dv')
#df.to_parquet(sp)

#df = local_variable_mean(df, 'phi_diff')
#df.to_parquet(sp)

#df = local_variable_mean(df, 'dvz')
#df.to_parquet(sp)

df = local_variable_mean(df, 'dvy')
df.to_parquet(sp)