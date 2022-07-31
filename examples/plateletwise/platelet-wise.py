from plateletanalysis.variables.plateletwise import construct_platelet_df
import pandas as pd
import os
from plateletanalysis.variables.measure import path_disp_n_tortuosity

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
p = os.path.join(d, '211206_saline_df_220614-amp0.parquet')
df = pd.read_parquet(p)

# re-add incorrect var
#df = path_disp_n_tortuosity(df)
#df['terminating'] = df['nrtracks'] == df['tracknr']
#df.to_parquet(sp)

# REMOVE OUTLIER FILE
#df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3']

# REMOVE SHORT TRACKS 
df = df[df['nrtracks'] > 10]

# Add stability
nrterm = df['nrterm'].values
n_remain = df['frame'].max() - df['frame'].values
df['stability'] = nrterm / n_remain
df = df.dropna(subset=['stability'])
df = df[df['stability'] < 2]

# New save name
sp = os.path.join(d, 'plateletwise_211206_saline_df_220614-amp0.parquet')


# Variables
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
# Construct the platelet aggregate df
pdf = construct_platelet_df(df, save=sp, cols=cols)

#import napari

#paths = pd.unique(df['path'])
#v = napari.Viewer()
#for p in paths:
 #   pdf = df[df['path'] == p]
  #  tracks = pdf[['particle', 'frame', 'zs', 'ys', 'x_s']].values
   # v.add_tracks(tracks, name=p, properties=pdf, visible=False)