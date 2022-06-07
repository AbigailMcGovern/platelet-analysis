import pandas as pd
import os
from plateletanalysis.analysis.clustering import plot_clusters, umap_dbscan_cluster


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
ssp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_f171-193.parquet')
df = pd.read_parquet(ssp)
eps_list=(0.1, 0.15, .25)
df = umap_dbscan_cluster(df, 'SCnV', save=ssp, eps_list=eps_list)
cluster_cols = [f'SCnV_udbscan_{e}'  for e in eps_list]
umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
plot_clusters(df, cluster_cols[0], umap_cols=umap_cols)