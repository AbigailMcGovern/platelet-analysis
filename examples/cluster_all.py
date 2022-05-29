import pandas as pd
import os
from plateletanalysis.analysis.clustering import cluster_spherical_coordinates, plot_clusters, umap_dbscan_cluster


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'

files = ['211206_cang_df_spherical-coords.parquet',
         '211206_saline_df_spherical-coords.parquet', 
         '211206_biva_df_spherical-coords.parquet', 
         '211206_ctrl_df_spherical-coords.parquet', 
         '211206_sq_df_spherical-coords.parquet', 
         '211206_veh-sq_df_spherical-coords.parquet']

# concatenate and save
#df = None
#for f in files:
#    sp = os.path.join(d, f)
#    if df is None:
#        df = pd.read_parquet(sp)
#    else:
#        n_df = pd.read_parquet(sp)
#        df = pd.concat([df, n_df.copy()], axis=0)
#        del n_df
#print(pd.unique(df['inh']))
#sp = os.path.join(d, '211206_cang_saline_biva_ctrl_sq_vehsq_df.parquet')
#df.to_parquet(sp)
#df = pd.read_parquet(sp)
#df0 = df[df['inh'] == '_cang_']
#df1 = df[df['inh'] == '_saline_']
#df = pd.concat([df0, df1])
#df = df.reset_index(drop=True)
#df = df[df['nrtracks'] > 4]
#print(pd.unique(df['inh']))
#sp = os.path.join(d, '211206_cang_saline_df.parquet')

#sp = os.path.join(d, '211206_cang_saline_df_cleaned.parquet')
sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
#sp = os.path.join(d, '211206_saline_df_spherical-coords_clust-cleaned.parquet')
#sp = os.path.join(d, '211206_cang_df_cleaned.parquet')
#df.to_parquet(sp)
#sp = os.path.join(d, '211206_saline_df_spherical-coords_220522-0.parquet')
df = pd.read_parquet(sp)

# REMOVE OUTLIER FILE
df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3']
sp = os.path.join(d, '211206_saline_df_spherical-coords_220527_0.parquet')


# umap dbscan and save 
eps_list=(0.1, 0.15)
#cols = ('rho', 'theta', 'phi', 'rho_diff', 'theta_diff', 'phi_diff', 'frame', 'nb_density_15', 'nrterm') #, 'path_len', 'disp', 'ca_corr')
df = umap_dbscan_cluster(df, 'SCnV_220527', save=sp, eps_list=eps_list) #, cols=cols)#, umap_cols=umap_cols, min_samples=5) #, cols=cols)

# plot the results
umap_cols = ['umap_0_SCnV_220527', 'umap_1_SCnV_220527']
cluster_cols = [f'SCnV_220527_udbscan_{e}' for e in eps_list]
plot_clusters(df, cluster_cols[1], umap_cols=umap_cols) #, frac=.05)