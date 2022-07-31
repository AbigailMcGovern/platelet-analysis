
import napari
import numpy as np
import pandas as pd
from nd2_dask.nd2_reader import nd2_reader


def unify_pixel_and_tracks(df, path_df):
    # this code works but fundementally isn't useful because most of the files 
    # were re-tracked and the pixel information removed
    path_df = path_df.rename(columns={'t' : 'frame', 'file': 'path'})
    new_df = path_df[['path', 'particle', 'frame', 'z_pixels', 'y_pixels', 'x_pixels']]
    new_df['idx'] = new_df['path'] + '_' + new_df['particle'].apply(str) + '_' + new_df['frame'].apply(str) 
    new_df = new_df.set_index('idx')
    idx_0 = new_df.index.values[0]
    path = str(new_df.loc[idx_0, 'path']) # the path df only has one unique value for path 
    df = df[df['path'] == path]
    df['idx'] = df['path'] + '_' + df['particle'].apply(str) + '_' + df['frame'].apply(str) 
    df = df.set_index('idx')
    new_df = new_df.drop(['path', 'particle', 'frame'], axis=1)
    new_df = pd.concat([df, new_df], axis=1)
    new_df = new_df[new_df['nrtracks'] > 4]
    return new_df
# need to match the path and particle ID in the original and the worker df
# probably generate a new df with 


def display_tracks_and_properties(df, images, scale=(1, 4, 1, 1)):
    tracks = get_tracks(df)
    viewer = napari.Viewer()
    for image in images:
        viewer.add_image(image, scale=scale)
    viewer.add_tracks(tracks, scale=scale, properties=df)
    viewer.add_points(tracks[:, 1:], scale=scale, size=2)
    napari.run()


def display_track_clusters(df, images, cluster_col, scale=(1, 4, 1, 1)):
    viewer = napari.Viewer()
    for image in images:
        viewer.add_image(image, scale=scale, color='red')
    clusters = pd.unique(df[cluster_col])
    for c in clusters:
        c_df = df[df[cluster_col] == c]
        tracks = get_tracks(df)


def display_all_tracks(df):
    files = pd.unique(df['path'])
    v = napari.Viewer()
    for f in files:
        f_df = df[df['path'] == f]
        tracks = get_tracks(f_df, ('particle', 'frame', 'zs', 'ys', 'x_s'))
        v.add_tracks(tracks, properties=f_df, name=f, visible=False)
    napari.run()




# ----------------
# Helper functions
# ----------------

def get_tracks(df, cols=('particle', 'frame', 'z_pixels', 'y_pixels', 'x_pixels')):
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


