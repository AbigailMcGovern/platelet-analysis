from old_modules.scatter import multi_basic_scatter_save
from plateletanalysis.analysis.clustering import plot_clusters
import pandas as pd
import os

d = '/fs02/rl54/dataframes'
path = os.path.join(d, '/fs02/rl54/dataframes/211206_saline_df_spherical-coords_220531_185654.parquet')
df = pd.read_parquet(path)
dt = '220531_185654'
umap_cols = ['umap_0_SCnV', 'umap_1_SCnV']
eps_list=(0.1, 0.15, 0.2)
cluster_cols = [f'SCnV_udbscan_{e}' for e in eps_list]
save_dir = f'/fs02/rl54/clustering/{dt}_UMAP-DBSCAN_SCnV'

for col in cluster_cols:
    n = dt + '_' + col + '_scatter.pdf'
    sp = os.path.join(save_dir, n)
    plot_clusters(df, col, umap_cols, save=sp, show=False, size=(10, 10))


cols = ['rho', 'theta', 'phi', 'rho_diff', 'theta_diff', 'phi_diff', 'frame', 'nb_density_15', 'nrterm', 
        'path_len', 'disp', 'ca_corr', 'zs', 'x_s', 'ys', 'elong', 'c1_mean', 'stab', 'nrtracks', 'tracknr']

multi_basic_scatter_save(df, umap_cols[0], umap_cols[1], cols, save_dir, f'UMAP-DBSCAN_SCnV')

cols = ['path', 'particle', 'rho', 'theta', 'phi', 'rho_diff', 'theta_diff', 'phi_diff', 'frame', 'nb_density_15', 'nrterm', 
        'path_len', 'disp', 'ca_corr', 'zs', 'x_s', 'ys', 'elong', 'c1_mean', 'stab', 'nrtracks', 'tracknr']