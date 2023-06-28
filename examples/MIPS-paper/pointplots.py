import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from plateletanalysis import add_region_category, tri_phase_var, add_basic_variables_to_files


MIPS_order = ['DMSO (MIPS)', 'MIPS']
cang_order = ['saline','cangrelor']#['Saline','Cangrelor','Bivalirudin']
SQ_order = ['DMSO (SQ)', 'SQ']
pal_MIPS  = dict(zip(MIPS_order, sns.color_palette('Blues')[2::3]))
pal_cang = dict(zip(cang_order, sns.color_palette('Oranges')[2::3]))
pal_SQ = dict(zip(SQ_order, sns.color_palette('Greens')[2::3]))
pal1={**pal_MIPS,**pal_cang,**pal_SQ}


def first_frames_pointplot(
        df, 
        vars, 
        treatment='MIPS', 
        control='DMSO (MIPS)',  
        ):
    df = add_region_category(df)
    df = tri_phase_var(df)
    print(df.head())
    id_vars = ['path', 'tri_fsec', 'region', 'tracknr']
    dfg = df[(df.inh == treatment) & (df.tracknr < 6)].groupby(id_vars)[vars].mean().reset_index()
    dfg_cmeans = df[(df['treatment'] == control) & (df['tracknr'] < 6)].groupby(id_vars[1:])[vars].mean().reset_index()
    df_m = (dfg.set_index(['tri_fsec', 'region', 'tracknr', 'path'])[vars]/dfg_cmeans.set_index(['tri_fsec', 'region', 'tracknr'])[vars]) * 100 #.reset_index()
    df_m = df_m.reset_index()
    df_melt = df_m.melt(id_vars = id_vars[1:], value_vars = vars)
    # ALL THROMBI 
    g = sns.catplot(data = df_melt[df_melt.tracknr > 1], y='value', x='tracknr', hue='region', col='tri_fsec', row='variable',
                height=2, aspect=1, kind='point', sharey='row', errorbar="se", palette='viridis')
    g.set_titles("{col_name} \n {row_name} ")
    g.despine(top=True, right=True, left=True, bottom=True, offset=None, trim=False)
    plt.show()


d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet', 
              '211206_cang_df.parquet', '211206_saline_df_220827_amp0.parquet', 
              '211206_sq_df.parquet', '211206_veh-sq_df.parquet', '230301_MIPS_and_DMSO.parquet')
file_paths = [os.path.join(d, n) for n in file_names]

df = add_basic_variables_to_files(file_paths, density=True)
vars = ['nba_d_5', 'stab', 'ca_corr']
first_frames_pointplot(df, vars)