from plateletanalysis.variables.neighbours import embolysis_proximity
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_220612-amp2.parquet')
df = pd.read_parquet(sp)


#paths = pd.unique(df['path'])
#df = df[df['path'] == paths[0]]
df = embolysis_proximity(df)
sp = os.path.join(d, '211206_saline_df_220613-amp0.parquet')
df.to_parquet(sp)