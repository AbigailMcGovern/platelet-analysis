import pandas as pd
import os
from plateletanalysis.analysis.clustering import umap_dbscan_cluster, plot_clusters
from old_modules.scatter import multi_basic_scatter_save
from datetime import datetime


d = '/fs02/rl54/dataframes'
path = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(path)

now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")


df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3'] # this clot looks very different and is likely to confuse clustering

# clean the data
df = df[df['nrtracks'] > 4]
#df = df[(df['phi_diff'] > -0.0825) & (df['phi_diff'] < 0.0825)]
#df = df[(df['theta_diff'] > -0.0407) & (df['theta_diff'] < 0.0407)]

# get only the unstable platelets
if 'terminating' not in df.columns.values:
        df['terminating'] = df['tracknr'] == df['nrtracks']
df = df[df['terminating'] == True]
df = df[df['frame'] < 193]
# save clustering output here
sp = os.path.join(d, f'211206_saline_df_spherical-coords_SC-emb_{dt}.parquet')

eps_list=(0.1, 0.15, 0.2)
cols = ('rho', 'theta', 'phi', 'frame') #, 'path_len', 'disp', 'ca_corr')
df = umap_dbscan_cluster(df, 'SC-emb', save=sp, eps_list=eps_list, cols=cols)
df.to_parquet(sp)

# save plots here
save_dir = f'/fs02/rl54/clustering/{dt}_UMAP-DBSCAN_SC-emb'
os.makedirs(save_dir, exist_ok=True)

# build and save the cluster plots
umap_cols = ['umap_0_SC-emb', 'umap_1_SC-emb']
cluster_cols = [f'SC-emb_udbscan_{e}' for e in eps_list]
for col in cluster_cols:
    n = dt + '_' + col + '_scatter.pdf'
    sp = os.path.join(save_dir, n)
    plot_clusters(df, col, umap_cols, save=sp, show=False, size=(10, 10))


cols = ('rho', 'theta', 'phi', 'rho_diff', 'theta_diff', 'phi_diff', 'frame', 'nb_density_15', 'nrterm', 
        'path_len', 'disp', 'ca_corr', 'zs', 'x_s', 'ys', 'elong', 'c1_mean', 'stab', 'nrtracks', 'tracknr')


# buid and save the variable colour coded UMAP plots
multi_basic_scatter_save(df, umap_cols[0], umap_cols[1], cols, save_dir, f'UMAP-DBSCAN_SC-emb')





