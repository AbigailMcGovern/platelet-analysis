import pandas as pd
import os
from plateletanalysis.analysis.clustering import cluster_spherical_coordinates, plot_clusters, umap_dbscan_cluster

#d = ('/Users/amcg0011/Data/platelet-analysis/dataframes')
#sp = os.path.join(d, '211206_saline_df_nb_dens.parquet')
#df = pd.read_parquet(sp)
#df = df[df['path'] == '200527_IVMTR73_Inj4_saline_exp3']

#sp = os.path.join(d, '211206_saline_df_nb_dens_demo.parquet')
#df = pd.read_parquet(sp)
#df = cluster_spherical_coordinates(df, save_checkpoint=sp)
#df.to_parquet(sp)
#df = plot_clusters(df, 'dbscan_1', umap_cols=['umap_0_scoords', 'umap_1_scoords'], frame_range=(100, 110))
#df.to_parquet(sp)

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
#sp = os.path.join(d, '211206_saline_df_nb_dens.parquet')
#df = pd.read_parquet(sp)
#df = df[df['frame'] < 10]
#cluster_spherical_coordinates(df, save_checkpoint=sp)
#sp = os.path.join(d, '211206_saline_df_nb_dens_f0-10.parquet')
#df.to_parquet(sp)

#df = pd.read_parquet(sp)
#df = cluster_spherical_coordinates(df, save_checkpoint=sp)
#df = plot_clusters(df, 'dbscan_5', umap_cols=['umap_0_scoords', 'umap_1_scoords'])
#df.to_parquet(sp)

#print(df['dbscan_5'].value_counts())
#sp = os.path.join(d, '211206_saline_df_f0-10_UD-cluster.parquet')
#df = pd.read_parquet(sp)
#df = umap_dbscan_cluster(df, 'SCnV', save=sp)

#umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
#cluster_cols = ['SCnV_udbscan_1', 'SCnV_udbscan_3', 'SCnV_udbscan_5', 'SCnV_udbscan_10']
#plot_clusters(df, cluster_cols[0], umap_cols=umap_cols)#

# removed outliers and untracked objects in ipython 

#sp = os.path.join(d, '211206_saline_df_f0-10_cleaned_UD-cluster.parquet')
#df = pd.read_parquet(sp)
#umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
#eps_list=(0.1, 0.15, .25, 0.5)
#df = umap_dbscan_cluster(df, 'SCnV', save=sp, eps_list=eps_list, min_samples=10, umap_cols=umap_cols)
#cluster_cols = [f'SCnV_udbscan_{e}'  for e in eps_list]
#cluster_cols = ['SCnV_udbscan_0.1', 'SCnV_udbscan_0.25', 'SCnV_udbscan_0.5', 'SCnV_udbscan_0.75', 'SCnV_udbscan_1']
#plot_clusters(df, cluster_cols[0], umap_cols=umap_cols)#
#plot_clusters(df, cluster_cols[1], umap_cols=umap_cols)#


#sp = os.path.join(d, '211206_saline_df_nb_dens.parquet')
#df = pd.read_parquet(sp)
#sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan.parquet')
#df = pd.read_parquet(sp)
#eps_list=(0.1, 0.15, .25, 0.5)
#umap_dbscan_cluster(df, 'SCnV', save=sp, eps_list=eps_list) #min_samples=10, umap_cols=umap_cols)]
#cluster_cols = [f'SCnV_udbscan_{e}'  for e in eps_list]
#umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
#plot_clusters(df, cluster_cols[2], umap_cols=umap_cols) # removed fluff in ipythondf[df[]]

#sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
#df = pd.read_parquet(sp)
#df = df[df['nrtracks'] > 4]
#sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_ALL.parquet')
#eps_list=(0.1, 0.15, .25)
#df = umap_dbscan_cluster(df, 'SCnV', save=sp, eps_list=eps_list)
#cluster_cols = [f'SCnV_udbscan_{e}'  for e in eps_list]
#umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
#plot_clusters(df, cluster_cols[0], umap_cols=umap_cols, frac=0.05)

#sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_ALL_cleaned.parquet')
#df = pd.read_parquet(sp)
#eps_list=(0.1, 0.15, .25)
#df = umap_dbscan_cluster(df, 'SCnV', save=sp, eps_list=eps_list)
#cluster_cols = [f'SCnV_udbscan_{e}'  for e in eps_list]
#umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
#plot_clusters(df, cluster_cols[0], umap_cols=umap_cols, frac=0.05)

#sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_sub-cluster.parquet')
#df = pd.read_parquet(sp)
#eps_list=(0.1, 0.15, .25)
#df = umap_dbscan_cluster(df, 'SCnV-sub', save=sp, eps_list=eps_list)
#cluster_cols = [f'SCnV-sub_udbscan_{e}'  for e in eps_list]
#umap_cols = ['umap_0_SCnV-sub', 'umap_1_SCnV-sub']
#plot_clusters(df, cluster_cols[0], umap_cols=umap_cols, frac=0.05)

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(sp)
eps_list=(0.1, 0.15, .25)
df = umap_dbscan_cluster(df, 'SCnV-sub', save=sp, eps_list=eps_list)
cluster_cols = [f'SCnV-sub_udbscan_{e}'  for e in eps_list]
umap_cols = ['umap_0_SCnV-sub', 'umap_1_SCnV-sub']
plot_clusters(df, cluster_cols[0], umap_cols=umap_cols, frac=0.05)

# 'rho_diff', 'theta_diff', 'rho', 'theta', 'phi', 'phi_diff'
# 'nb_density_15', 'nrterm', 'tracknr', 
# 'path_len', 'disp', 'ca_corr

cols = ['rho', 'theta', 'phi', 'rho_diff', 'theta_diff', 'phi_diff', 'nb_density_15', 'nrterm', 'tracknr', 'path_len', 'disp', 'ca_corr']
