from plateletanalysis.io import get_experiment_df, add_info_from_file_name

#C2aKD_p = '/Users/amcg0011/Data/platelet-analysis/C2aKD_tracks/C2aKD_local_platelet-tracks'
C2aCtrl = '/Users/amcg0011/Data/platelet-analysis/Ca2_control_local_platelet-tracks'
ps = '/Users/amcg0011/Data/platelet-analysis/211020_IVM-TR139_CD62p_local_platelet-tracks'

# get the experiment platelet df and file metadata df
df, meta_df = get_experiment_df(ps, 'p-selectin', tx_col='cohort')

# get experiment info from file name
df = add_info_from_file_name(df, path_col='path')
meta_df = add_info_from_file_name(meta_df)
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220214_P-selectin_df.parquet'
df.to_parquet(save)
save = '/Users/amcg0011/Data/platelet-analysis/dataframes/220214_P-selectin_meta.parquet'
meta_df.to_parquet(save)