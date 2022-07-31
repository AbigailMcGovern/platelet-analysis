from plateletanalysis.variables.neighbours import local_calcium
import pandas as pd
import os

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sn = '211206_saline_df_220612-amp0.parquet'
sp = os.path.join(d, sn)
df = pd.read_parquet(sp)

df = local_calcium(df)

s = os.path.join(d, '211206_saline_df_220612-amp1.parquet')
df.to_parquet(s)