import pandas as pd
import os
from plateletanalysis import add_basic_variables_to_files
from plateletanalysis.analysis.summary_data import experiment_region_epoch_data

# ------------------
# Get data from file
# ------------------
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]

df = add_basic_variables_to_files(file_paths, density=True)
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_exp_region_epoch_data.csv'

result = experiment_region_epoch_data(df, save_path)