from plateletanalysis.topology.donutness import largest_loop_comparison_data, plot_donut_comparison
import pandas as pd
import numpy as np
import os


# -----------------
# Obtain donut data
# -----------------

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
names = ['211206_saline_df_220827_amp0.parquet', '211206_biva_df.parquet', 
         '211206_cang_df.parquet', '211206_sq_df.parquet', 
         '211206_mips_df_220818.parquet', '211206_ctrl_df.parquet', 
         '211206_par4--_df.parquet', '211206_par4--biva_df.parquet', 
         '211206_salgav_df.parquet', '211206_salgav-veh_df.parquet', 
         '211206_veh-sq_df.parquet', '211206_veh-mips_df.parquet'] # 12 dataframes 
paths = [os.path.join(d, n) for n in names]
#sp = os.path.join(d, 'short_mips.parquet') 
#paths = [sp, ] # only 2 thrombi for debugging
sd = '/Users/amcg0011/Data/platelet-analysis/TDA/treatment_comparison'
save_path = os.path.join(sd, '221025_longest-loop-analysis.csv')
data = largest_loop_comparison_data(paths, save_path)



