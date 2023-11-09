from plateletanalysis.topology.donutness import donutness_data
from plateletanalysis.variables.measure import quantile_normalise_variables_frame
from plateletanalysis.variables.neighbours import add_neighbour_lists, local_density
import pandas as pd
import os


sn = '211206_veh-sq_df_spherical-coords.parquet'
name = 'DMSO(SQ)'
sd = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
df_p = os.path.join(sd, sn)
df = pd.read_parquet(df_p)
if 'nb_density_15' not in df.columns.values:
    df = add_neighbour_lists(df)
    df = local_density(df)
    df.to_parquet(df_p)
if 'nb_density_15_pcntf' not in df.columns.values:
    df = quantile_normalise_variables_frame(df)
    df.to_parquet(df_p)
df = df[df['nrtracks'] > 1]
out = donutness_data(df, units='AU')
sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/data'
sp = os.path.join(sd, f'{name}_donut_data_scaled_sn200_n100_c50_gt1tks.csv')
out.to_csv(sp)

# -------
# Options
# -------
sn = '211206_veh-sq_df_spherical-coords.parquet'
name = 'DMSO(SQ)'

sn = '211206_sq_df.parquet'
name = 'SQ'

sn = '211206_ctrl_df.parquet'
name = 'control'

sn = '211206_veh-mips_df.parquet'
name = 'DMSO(MIPS)'

sn = '211206_mips_df.parquet'
name = 'MIPS'

sn = '230301_MIPS_and_DMSO.parquet'
name = 'DMSO(MIPS)_and_MIPS_2023'

sn = '211206_biva_df.parquet'
name = 'bivalirudin'

sn = '211206_cang_df.parquet'
name = 'cangrelor'

sn = '211206_par4--_df.parquet'
name = 'PAR4--'

sn = '211206_salgav_df.parquet'
name = 'saline_gavarge'

sn = '211206_salgav-veh_df.parquet'
name = 'saline_gavarge_vehicle'

#...