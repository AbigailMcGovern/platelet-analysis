from plateletanalysis.analysis.stats import cross_corr
import pandas as pd
import os


d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, 'plateletwise_211206_saline_df_220614-amp0.parquet')
df = pd.read_parquet(sp)


vars = ['ys', 'zs', 'dv', 'cont', 'cont_p', 'depth', 'tracknr',
       'nrtracks', 'cont_tot','ca_corr', 'rho', 
       'theta', 'phi', 'nb_density_15', 'nrterm', 'disp', 'rho_diff', 
       'phi_diff', 'theta_diff', 'nd15_percentile', 'fibrin_dist', 
       'fibrin_cont', 'phi_pcnt','phi_diff_pcnt', 'rho_pcnt', 
       'theta_pcnt','zs_pcnt', 'fibrin_dist_pcnt', 'nb_ca_corr_15', 
       'nb_ca_diff_15_diff','nb_ca_copying_15', 'nb_cont_15', 'stability']

mean = ['mean_' + v for v in vars]
std = ['std_' + v for v in vars]
start = ['start_tracknr', ]
cols = mean + std + start

r = '/Users/amcg0011/PhD/Platelets/Results/Correlations'
save = os.path.join(r, 'plateletwise_211206_saline_df_220614-amp0_pearson-corr.csv')

cross_corr(df, cols, save)