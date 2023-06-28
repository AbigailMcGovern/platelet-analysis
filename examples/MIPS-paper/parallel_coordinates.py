import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from plateletanalysis import add_basic_variables_to_files
# function to be tested
from plateletanalysis import region_parallel_coordinate_data
from plateletanalysis.analysis.plots import pal1
import plotly.express as px


def _colour_from_pal(tx):
    c = pal1[tx]
    return c

# ------------------
# Get data from file
# ------------------
d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]

#df = add_basic_variables_to_files(file_paths)
psp = f'/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_3/230420_count-and-growth-pcnt_peaks.csv'
peaks = pd.read_csv(psp)
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/Figure_4/230502_parallel_coordinate_plot_data.csv'

#result = region_parallel_coordinate_data(df, peaks, save_path)
result = pd.read_csv(save_path)
treatments = ('MIPS', 'SQ', 'cangrelor')
controls = ('DMSO (MIPS)', 'DMSO (SQ)', 'saline')
vars = ('platelet count', 'platelet density (um^-3)', 
                   'stability', 'recruitment (s^-1)', 
                   'P(recruited < 15 s)', 'sliding (ums^-1)')
for k, grp in result.groupby('treatment'):
    print(k)
    print(pd.unique(grp.path))
result = result[result['phase'] == 'consolidation']
result['treatment colour'] = result.treatment.apply(_colour_from_pal)
for i, tx in enumerate(treatments):
    if tx =='MIPS':
        data = pd.concat([result[result['treatment'] == tx], result[result['treatment'] == controls[i]]])
        for v in vars:
            pd.plotting.parallel_coordinates(data, 'treatment',
                                              [f'anterior: {v}', 
                                               f'center: {v}', 
                                               f'lateral: {v}', 
                                               f'posterior: {v}'], 
                                               colormap='Set1')
            df_ = data.copy()
            # color     : Values from this column are used to assign color to the poly lines.
            # dimensions: Values from these columns form the axes in the plot.
            #fig = px.parallel_coordinates(df_, color="treatment colour", 
            #                              dimensions=[f'anterior: {v}', 
            #                               f'center: {v}', 
            #                               f'lateral: {v}', 
            #                               f'posterior: {v}'])
            plt.show()
            plt.close()
            #fig.show()
