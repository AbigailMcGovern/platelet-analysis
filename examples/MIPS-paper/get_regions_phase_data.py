import pandas as pd
import os
from plateletanalysis import add_basic_variables_to_files
from plateletanalysis import experiment_region_phase_data


# ------------------
# Get data from file
# ------------------
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]

df = add_basic_variables_to_files(file_paths, density=True)
#psp = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230420_count-and-growth-pcnt_peaks.csv'
psp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_max_count_and_time.csv'
peaks = pd.read_csv(psp)
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_exp_region_phase_data.csv'

result = experiment_region_phase_data(df, peaks, save_path)