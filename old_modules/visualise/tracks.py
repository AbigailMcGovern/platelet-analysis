
import napari
import numpy as np
import pandas as pd
from nd2_dask.nd2_reader import nd2_reader



def display_all_tracks(df, id_t_z_y_x=('particle', 'frame', 'zs', 'ys', 'x_s')):
    files = pd.unique(df['path'])
    v = napari.Viewer()
    for f in files:
        f_df = df[df['path'] == f]
        tracks = get_tracks(f_df, id_t_z_y_x)
        v.add_tracks(tracks, properties=f_df, name=f, visible=False)
    napari.run()


# ----------------
# Helper functions
# ----------------

def get_tracks(df, cols=('frame', 'zs', 'ys', 'x_s')):
    tracks = df[list(cols)].values
    return tracks


def read_ND2(path):
    layerlist = nd2_reader(path)
    images = []
    for i, layertuple in enumerate(layerlist):
        images.append(layertuple[0]) # a dask array
        #channel = layertuple[1]['name']
        #if '647' in channel:
         #   image = layertuple[0]
          #  data = layertuple[1]
    return images


