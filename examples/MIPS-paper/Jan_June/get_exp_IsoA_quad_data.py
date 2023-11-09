import pandas as pd
import os
import numpy as np
from plateletanalysis import add_basic_variables_to_files
from plateletanalysis.analysis.mips_analysis import experiment_quadrant_isoA_phase_data


# ------------------
# Get data from file
# ------------------
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]


df = add_basic_variables_to_files(file_paths, density=True)
psp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_max_count_and_time.csv'
peaks = pd.read_csv(psp)


# calc
injs = ['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']
df.loc[df.path.isin (injs), 'ca_corr'] = np.nan


save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230623_exp_cylrad_binned_phase_data.csv'
res = experiment_quadrant_isoA_phase_data(df, peaks, save_path)

