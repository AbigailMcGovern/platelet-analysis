from plateletanalysis import cylrad_binned_region_phase_data, add_basic_variables_to_files
import pandas as pd
import os

d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]

df = add_basic_variables_to_files(file_paths, density=True)
psp = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230420_count-and-growth-pcnt_peaks.csv'
peaks = pd.read_csv(psp)

save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_4/230509_cylrad_binned_phase_data.csv'

res = cylrad_binned_region_phase_data(df, peaks, save_path)