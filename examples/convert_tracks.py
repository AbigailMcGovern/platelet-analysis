from plateletanalysis.io import get_experiment_df, add_info_from_file_name
from plateletanalysis.measure import finite_difference_derivatives,\
    nearest_neighbours_average, stability
from plateletanalysis.transform import z_floor

C2aKD_p = '/Users/amcg0011/Data/platelet-analysis/C2aKD_tracks/C2aKD_local_platelet-tracks'

# get the experiment platelet df and file metadata df
C2aKD_df, meta_df = get_experiment_df(C2aKD_p, 'C2aKD')

# get experiment info from file name
C2aKD_df = add_info_from_file_name(C2aKD_df, path_col='path')
meta_df = add_info_from_file_name(meta_df)
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220214_C2aKD_df.parquet'
C2aKD_df.to_parquet(save)
print('saved checkpoint: added info from file names')

# discreet spatial difference derivatives (approximation for velocity)
C2aKD_df = finite_difference_derivatives(C2aKD_df)
C2aKD_df.to_parquet(save)
print('saved checkpoint: added derivatives')

# average nearest neighbour distance for 5, 10, and 15 neighbours
C2aKD_df = nearest_neighbours_average(C2aKD_df)
C2aKD_df.to_parquet(save)
print('saved checkpoint: added n dist')

# stability measure
C2aKD_df = stability(C2aKD_df)
C2aKD_df.to_parquet(save)
print('saved checkpoint: added stab')

# z floor
C2aKD_df = z_floor(C2aKD_df)
C2aKD_df.to_parquet(save)
print('saved checkpoint: added z floor')

# save the data frame
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220214_C2aKD_df.parquet'
C2aKD_df.to_parquet(save)

