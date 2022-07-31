from plateletanalysis.variables.measure import fibrin_cotraction
import pandas as pd
import os

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
dfp = os.path.join(d, '211206_saline_df_220610.parquet')
df = pd.read_parquet(dfp)

df = fibrin_cotraction(df)
sp = os.path.join(d, '211206_saline_df_220611-0.parquet')
df.to_parquet(sp)