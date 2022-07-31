from plateletanalysis.variables.neighbours import local_calcium_diff_and_copy
import pandas as pd
import os

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_220612-amp1.parquet')
df = pd.read_parquet(sp)

#paths = pd.unique(df['path'])
#df = df[df['path'] == paths[0]]
df = local_calcium_diff_and_copy(df)

s = os.path.join(d, '211206_saline_df_220612-amp2.parquet')
df.to_parquet(s)