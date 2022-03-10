import plateletanalysis

C2aKD_p = '/Users/amcg0011/Data/platelet-analysis/C2aKD_tracks/C2aKD_local_platelet-tracks'

df, meta_df = plateletanalysis.get_experiment_df(C2aKD_p, 'C2aKD')
df = plateletanalysis.adjust_coordinates(df, meta_df)

def fix_names(s: str):
    new = s.replace(' ', '')
    new = new.replace('-', '')
    return new

df['path'] = df['path'].apply(fix_names)
df = plateletanalysis.add_info_from_file_name(df, positions=(0, 1, 3, 2, 4))
df = plateletanalysis.finite_difference_derivatives(df)
df = plateletanalysis.z_floor(df)
df = plateletanalysis.stability(df)
df = plateletanalysis.average_neighbour_distance(df)
df = plateletanalysis.contractile_motion(df)
df = plateletanalysis.point_depth(df)
df = plateletanalysis.add_basic_variables(df)

save_csv = '/Users/amcg0011/Data/platelet-analysis/dataframes/220214_C2aKD_df.csv'
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220303_C2aKD_df.parquet'
save_csv = '/Users/amcg0011/Data/platelet-analysis/dataframes/220303_C2aKD_df.csv'
# save the data frame
df.to_parquet(save)


import plateletanalysis
import pandas as pd
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220303_C2aKD_df.parquet'
df = pd.read_parquet(save)
