import pandas as pd
import os
from plateletanalysis.analysis.clustering import umap_dbscan_cluster, plot_clusters
from plateletanalysis.visualise.scatter import multi_basic_scatter_save
from datetime import datetime

d = '/fs02/rl54/dataframes'
n = '220604-1_plateletwise_211206_saline_df_spherical-coords.parquet'
path = os.path.join(d, n)
df = pd.read_parquet(path)
now = datetime.now()
dt = now.strftime("%y%m%d_%H%M%S")


df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3'] # this clot looks very different and is likely to confuse clustering


# save clustering output here
sp = os.path.join(d, f'220603_plateletwise_211206_saline_df_spherical-coords_{dt}.parquet')

eps_list=(0.1, 0.15, 0.2)
cols = (
    'start_frame', 
    'mean_rho', 
    'mean_theta', 
    'mean_phi', 
    'mean_rho_diff',
    'mean_theta_diff', 
    'mean_phi_diff', 
    'mean_dv',
    'var_rho', 
    'var_theta', 
    'var_phi', 
    'var_rho_diff', 
    'var_theta_diff', 
    'var_phi_diff', 
    'var_dv', 
    #'start_rho', 
    #'start_phi', 
    #'start_theta', 
    #'end_rho', 
    #'end_phi', 
    #'end_theta', 
    ) 

# remove outliers
for col in cols:
    std3 = df[col].std() * 3
    mean = df[col].mean()
    lbound = mean - std3
    ubound = mean + std3
    df = df[(df[col] > lbound) & (df[col] < ubound)]

df = umap_dbscan_cluster(df, 'SCnV_mv', save=sp, eps_list=eps_list, cols=cols)
df.to_parquet(sp)

# save plots here
save_dir = f'/fs02/rl54/clustering/{dt}_UMAP-DBSCAN_SCnV_mv'
os.makedirs(save_dir, exist_ok=True)

# build and save the cluster plots
umap_cols = ['umap_0_SCnV_mv', 'umap_1_SCnV_mv']
cluster_cols = [f'SCnV_mv_udbscan_{e}' for e in eps_list]
for col in cluster_cols:
    n = dt + '_' + col + '_scatter.pdf'
    sp = os.path.join(save_dir, n)
    plot_clusters(df, col, umap_cols, save=sp, show=False, size=(10, 10))


cols = (
    'start_frame', 
    'mean_rho', 
    'mean_theta', 
    'mean_phi', 
    'mean_rho_diff',
    'mean_theta_diff', 
    'mean_phi_diff', 
    'mean_dv',
    'var_rho', 
    'var_theta', 
    'var_phi', 
    'var_rho_diff', 
    'var_theta_diff', 
    'var_phi_diff', 
    'var_dv', 
    'start_rho', 
    'start_phi', 
    'start_theta', 
    'end_rho', 
    'end_phi', 
    'end_theta', 
    'var_ca_corr', 
    'mean_ca_corr'
    ) 


# buid and save the variable colour coded UMAP plots
multi_basic_scatter_save(df, umap_cols[0], umap_cols[1], cols, save_dir, f'UMAP-DBSCAN_SCnV_mv')
