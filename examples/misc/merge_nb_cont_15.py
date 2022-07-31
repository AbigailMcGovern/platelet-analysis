import pandas as pd
import os



d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
n1 = '211206_saline_df_220612-0.parquet'
p1 = os.path.join(d, n1)
p0 = os.path.join(d, '211206_saline_df_220612-amp2.parquet')


df0 = pd.read_parquet(p0)
df0 = df0.set_index(['path', 'particle', 'frame'], drop=False)
df1 = pd.read_parquet(p1)
df1 = df1.set_index(['path', 'particle', 'frame'], drop=False)
df1_idxs = df1.index.values
df1_val = df1['nb_cont_15'].values
df0.loc[df1_idxs, 'nb_cont_15'] = df1_val
df0.reset_index(drop=True)

s = os.path.join(d, '211206_saline_df_220614-amp0.parquet')
df0.to_parquet(s)


