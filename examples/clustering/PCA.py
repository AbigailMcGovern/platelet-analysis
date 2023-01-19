from plateletanalysis.analysis.clustering import PCA_objects, PCA_all, PCA_corr, cross_corr
from plateletanalysis.visualise.scatter import basic_scatter

import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
#sp = os.path.join(d, '211206_saline_df_SCnV-umap-dbscan_ALL_cleaned.parquet')
sp = os.path.join(d, '220604-1_plateletwise_211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(sp)
#df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3']
sp = os.path.join(d, '220604-1_plateletwise_211206_saline_df_spherical-coords_220604-PCA2.parquet')

cols=(
        'mean_phi', 
        'mean_theta', 
        'mean_rho', 
        'mean_phi_diff', 
        'mean_theta_diff', 
        'mean_rho_diff', 
        'mean_dv', 
        'mean_ca_corr', 
        'var_ca_corr', 
        'start_frame', 
        'end_frame', 
        'mean_nb_density_15',
        'var_nb_density_15',
        'start_nrtracks', 
    )

cols=(
        'mean_phi', 
        'mean_theta', 
        'mean_rho', 
        'mean_phi_diff', 
        'mean_theta_diff', 
        'mean_rho_diff', 
        'mean_x_s', 
        'mean_ys', 
        'mean_zs', 
        'mean_dv', 
        'mean_dvx', 
        'mean_dvy', 
        'mean_dvz', 
        'mean_ca_corr', 
        'mean_elong', 
        'mean_flatness', 
        'mean_c1_mean', 
        'mean_c2_mean', 
        'var_phi', 
        'var_theta', 
        'var_rho', 
        'var_phi_diff', 
        'var_theta_diff', 
        'var_rho_diff', 
        'var_x_s', 
        'var_ys', 
        'var_zs', 
        'var_dv', 
        'var_dvx', 
        'var_dvy', 
        'var_dvz', 
        'var_ca_corr', 
        'start_frame', 
        'start_rho', 
        'start_phi', 
        'start_theta', 
        'end_rho', 
        'end_phi', 
        'end_theta', 
        'end_path_len', 
        'end_disp', 
        'end_tort', 
        'end_frame', 
        'mean_nb_density_15',
        'var_nb_density_15',
        'mean_n_neighbours', 
        'var_n_neighbours', 
        'start_nrtracks', 
        'mean_stab', 
        'var_stab', 
    )

cols=(
        'mean_phi', 
        'mean_theta', 
        'mean_rho', 
        'mean_phi_diff', 
        'mean_theta_diff', 
        'mean_rho_diff', 
        'mean_dv', 
        'mean_ca_corr', 
        'mean_nb_density_15',
        'start_nrtracks', 
    )
#df, pca = PCA_objects(df, 'SCnV', cols=cols)
#df.to_parquet(sp)
#print(pca.explained_variance_ratio_)
#print(pca.noise_variance_)

#sdf = df.sample(frac=0.05)
#pcs_cols = ['PCA_0_SCnV', 'PCA_1_SCnV']
#basic_scatter(df, pcs_cols[0], pcs_cols[1], 'mean_phi')

#PCA_all(df, cols)

save = '/Users/amcg0011/PhD/Platelets/platelet_clustering/220605_pearcorr_10.csv'
#PCA_corr(df, cols, save)
cross_corr(df, cols, save)