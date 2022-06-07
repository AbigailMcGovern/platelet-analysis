from plateletanalysis.variables.plateletwise import construct_platelet_df
import pandas as pd
import os
from plateletanalysis.variables.measure import path_disp_n_tortuosity

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'
sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(sp)

# re-add incorrect var
#df = path_disp_n_tortuosity(df)
#df['terminating'] = df['nrtracks'] == df['tracknr']
#df.to_parquet(sp)

# REMOVE OUTLIER FILE
df = df[df['path'] != '200527_IVMTR73_Inj2_saline_exp3']

# REMOVE SHORT TRACKS 
df = df[df['nrtracks'] > 10]

# New save name
sp = os.path.join(d, '220604-1_plateletwise_211206_saline_df_spherical-coords.parquet')

# Construct the platelet aggregate df
pdf = construct_platelet_df(df, save=sp)

#import napari

#paths = pd.unique(df['path'])
#v = napari.Viewer()
#for p in paths:
 #   pdf = df[df['path'] == p]
  #  tracks = pdf[['particle', 'frame', 'zs', 'ys', 'x_s']].values
   # v.add_tracks(tracks, name=p, properties=pdf, visible=False)