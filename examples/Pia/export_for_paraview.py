from plateletanalysis.visualise.paraview import points_for_paraview
import pandas as pd
import os

example_inj = '200519_IVMTR69_Inj4_dmso_exp3' # DMSO SQ
dmso_inj = '210511_IVMTR105_Inj5_DMSO2_exp3'
mips_inj = '210520_IVMTR108_Inj3_MIPS_exp3'
d = '/Users/amcg0011/Data/platelet-analysis/dataframes' 
mips_n = '211206_mips_df_220818.parquet'
dmso_n = '211206_veh-mips_df_220831.parquet'
mpath = os.path.join(d, mips_n)
dpath = os.path.join(d, dmso_n)
df = pd.read_parquet(mpath)
sd = '/Users/amcg0011/Data/platelet-analysis/paraview/mips_manuscript'
points_for_paraview(df, mips_inj, sd, 'mips_example_for_contrast')
df = pd.read_parquet(dpath)
points_for_paraview(df, dmso_inj, sd, 'dmso_example_for_contrast')
points_for_paraview(df, example_inj, sd, 'dmso_example')