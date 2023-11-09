from plateletanalysis.analysis.peaks_analysis import growth_data, experiment_data_df
import os
import pandas as pd
from plateletanalysis.variables.basic import time_seconds, add_terminating, time_tracked_var

dfdir = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/'
file = '230301_MIPS_and_DMSO.parquet'
sp_0 = os.path.join(dfdir, file)

name = 'DMSO(MIPS)_and_MIPS_2023'
sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
sp_1 = os.path.join(sd, f'{name}_donut_data_scaled_sn200_n100_c50_gt1tk.csv')

df = pd.read_parquet(sp_0)
df = time_seconds(df)
df = add_terminating(df)
df = time_tracked_var(df)
df = df[df['nrtracks'] > 1]

ddf = pd.read_csv(sp_1)
ddf = time_seconds(ddf)

sn_2 = f'{name}_growth_data_gt1tk.csv'
sp_2 = os.path.join(sd, sn_2)
gdf = growth_data(df)
gdf.to_csv(sp_2)

sn_3 = f'{name}_summary_data_gt1tk.csv'
sp_3 = os.path.join(sd, sn_3)
experiment_data_df(df, gdf, ddf, sp_3, extra=True)