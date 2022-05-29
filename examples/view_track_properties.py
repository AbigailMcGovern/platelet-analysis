from plateletanalysis.visualise.tracks import display_tracks_and_properties, read_ND2
from plateletanalysis.variables.transform import revert_to_pixel_coords
import pandas as pd
import os

# images (calcium, fibrin, GPIIb, TD)
d = '/Users/amcg0011/Data/platelet-analysis/demo-data'
ip = os.path.join(d, '200527_IVMTR73_Inj4_saline_exp3.nd2')
images = read_ND2(ip)
images = images[:3] # remove the empty TD channel

# meta data
mp = os.path.join(d, '210920_141056_seg-track_200527_IVMTR73_Inj4_saline_exp3_seg-md.csv')
mdf = pd.read_csv(mp)
path = mdf.loc[0, 'file']

# computed data frame
d1 = '/Users/amcg0011/Data/platelet-analysis/dataframes'
dfp = os.path.join(d1, '211206_saline_df_SCnV-umap-dbscan_ALL_cleaned.parquet')
df = pd.read_parquet(dfp)
new_df = df[df['path'] == path]

# get the new unified data frame
# new_df = unify_pixel_and_tracks(df, tdf)

new_df = revert_to_pixel_coords(new_df, mdf)

nsp = os.path.join(d, '210920_141056_seg-track_200527_IVMTR73_Inj4_saline_exp3_for-vis.parquet')
new_df.to_parquet(nsp)

# look at the tracks with the newly computed properties
#display_tracks_and_properties(new_df, images, scale=(1, 4, 1, 1))
