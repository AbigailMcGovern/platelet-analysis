
from plateletanalysis.variables.measure import quantile_normalise_variables_frame
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
from plateletanalysis.variables.basic import get_treatment_name
import pandas as pd
import os
from pathlib import Path

#names = ['211206_biva_df.parquet', '211206_cang_df.parquet', 
#         '211206_ctrl_df.parquet', '230301_MIPS_and_DMSO.parquet', 
#         '211206_veh-mips_df.parquet', '211206_veh-sq_df_spherical-coords.parquet', 
#         '211206_mips_df.parquet', '211206_par4--_df.parquet', 
#         '211206_salgav-veh_df.parquet', '211206_salgav_df.parquet', 
#         '211206_sq_df.parquet']
names = ['211206_biva_df.parquet','211206_par4--_df.parquet', ]
sd = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'


for sn in names:
    df_p = os.path.join(sd, sn)
    df = pd.read_parquet(df_p)
    df = df.drop(columns=['nb_particles_15', 'nb_disp_15', 'nb_density_15', 'nb_density_15_pcntf', 'treatment'])
    df = df[df['nrtracks'] > 1]
    print(df.columns.values)
    df['treatment'] = df['path'].apply(get_treatment_name)
    if 'pid' not in df.columns.values:
        df['pid'] = list(range(len(df)))
    df = add_neighbour_lists(df)
    df = local_density(df)
    df_pn = os.path.join(sd, Path(df_p).stem + '_cleaned.parquet')
    df.to_parquet(df_pn)
    df = quantile_normalise_variables_frame(df)
    df.to_parquet(df_pn)

