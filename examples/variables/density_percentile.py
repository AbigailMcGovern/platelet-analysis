from unittest import result
from plateletanalysis.variables.measure import find_clot_percentile_structure, find_density_percentiles
import matplotlib.pyplot as plt
import os
import napari
import pandas as pd
import numpy as np

d = '/Users/amcg0011/Data/platelet-analysis/dataframes'

sp = os.path.join(d, '211206_saline_df_spherical-coords.parquet')
df = pd.read_parquet(sp)
#print(len(df))

#results = find_clot_percentile_structure(df)
#sp = os.path.join(d, '211206_saline_platelet_percentiles.csv')
#results.to_csv(sp)

#results = pd.read_csv(sp)

#paths = pd.unique(results['path'])
#r1 = results[results['path'] == paths[1]]
#r1 = r1[r1['frame'] == 100]

#ax = plt.axes(projection='3d')
#x = r1['x_s'].values
#y = r1['ys'].values
#z = r1['zs'].values
#c = r1['percentile'].values
#ax.scatter(x, y, z, c=c, cmap='viridis', linewidth=0.5)
#plt.show()


files = pd.unique(df['path'])
#df = df[df['path'] == files[1]]

save = os.path.join(d, '211206_saline_df_spherical-coords_density_pcnt.parquet')

df= find_density_percentiles(df)

df.to_parquet(save)


#v = napari.view_image(images[0], colormap='viridis')