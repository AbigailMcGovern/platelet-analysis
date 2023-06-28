import pandas as pd
import os
from plateletanalysis import add_basic_variables_to_files
from plateletanalysis.analysis.summary_data import quadrant_isoA_phase_data, quadrant_isoA_heatmap_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# ------------------
# Get data from file
# ------------------
#d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
#file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
#              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
#              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
#file_paths = [os.path.join(d, n) for n in file_names]
#df = add_basic_variables_to_files(file_paths, density=True)
#psp = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230612_max_count_and_time.csv'
#peaks = pd.read_csv(psp)
#
## calc
#injs = ['210520_IVMTR109_Inj2_DMSO_exp3', '210520_IVMTR109_Inj3_DMSO_exp3', '210520_IVMTR109_Inj4_DMSO_exp3', '210520_IVMTR109_Inj6_DMSO_exp3']
#df.loc[df.path.isin (injs), 'ca_corr'] = np.nan


# ----------------------
# Get initial data sheet
# ----------------------
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230618_isoA_region_phase_data.csv'
#save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/old_data/Figure_4/230529_isoA_region_phase_plot_data.csv'
#result = quadrant_isoA_phase_data(df, peaks, save_path)
result = pd.read_csv(save_path)


# ------------------------
# Alterations to variables
# ------------------------
fix = True
if fix:
    result['average platelet sliding (um)'] = result['average platelet sliding (um)'] / 0.32
    result = result[result['phase'] == 'consolidation']
    result['initial platelet velocity change (um/s)'] = - result['initial platelet velocity change (um/s)']
    result['total change in velocity (um/s)'] = - result['total change in velocity (um/s)']


# ---------
# Variables
# ---------
vars_dict = {
    'platelet count' : 'count', 
    'platelet average density (um^-3)' : 'density', 
    'platelet density gain (um^-3)' : 'density gain', 
    'average platelet instability' : 'instability', 
    #'average net platelet loss (/min)' : 'net platelet loss',
    'number lost' : 'net platelet loss',
    'average platelet tracking time (s)' : 'time tracked',
    'P(< 15s)' : 'P(< 15s)',
    'P(> 60s)' : 'P(> 60s)', 
    'recruitment' : 'recruitment', 
    'shedding' : 'shedding', 
    'P(recruited < 15 s)' : 'P(recruited < 15 s)', 
    'P(recruited > 60 s)': 'P(recruited > 60 s)', 
    'average platelet y velocity (um/s)' : 'y-axis velocity', 
    'total change in velocity (um/s)' : 'net decceleration', 
    'average platelet corrected calcium' : 'calcium',
    'average platelet elongation' : 'elongation', 
    'initial platlet density (um^-3)' : 'initial density', 
    'initial platelet instability' : 'initial instability',
    'initial platelet velocity change (um/s)' : 'initial decelleration', 
    'initial corrected calcium' : 'initial calcium'
}


# ------------
# Heatmap data 
# ------------
save_path = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/summary_data/230618_regions_isoA_heatmap_data_MIPS.csv'
#heatmap_df = quadrant_isoA_heatmap_data(result, save_path, vars_dict, group='MIPS')
heatmap_df = pd.read_csv(save_path)
heatmap_df = heatmap_df.drop(columns=('Unnamed: 0'))


# ------------
# Heatmap plot
# ------------
def regions_heatmap(df):
    fig, ax = plt.subplots(1, 1)
    df = df.set_index('quadrant x iso_A')
    sns.heatmap(data=df, annot=False, center=100, cmap='seismic', vmax=250, ax=ax)
    fig.set_size_inches(10, 8)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.2)
    plt.show()

#result = result.set_index('iso_A_bin')
#m = result[(result['treatment'] == 'MIPS') & (result['quadrant'] == 'posterior')]
#d = result[(result['treatment'] == 'DMSO (MIPS)') & (result['quadrant'] == 'posterior')]
#print(m['average net platelet loss (/min)'])
#print(d['average net platelet loss (/min)'])

regions_heatmap(heatmap_df)