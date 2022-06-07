from plateletanalysis.variables.neighbours import  add_neighbour_lists, local_density, local_contraction
import numpy as np
import pytest
import pandas as pd
import os
from pathlib import Path


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
pa_dir = dir_path.parents[0] # plateletanalysis directory

def read_data():
    data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.parquet') # path to test data 
    # test data is frames 0-1 for 191016_IVMTR12_Inj1_ctrl_exp3
    df = pd.read_parquet(data_path)
    return df


def test_add_neighbour_lists():
    df = read_data()
    df = add_neighbour_lists(df)
    # TODO:
    # add test once established, example:
    # assert...
    return df

def test_local_contraction():
    df = read_data()
    df = local_contraction(df)
    # assert df['nb_cont_15'].max() == 4.369167487664094
    return df


if __name__ == '__main__':
    import napari
    import zarr
    from scipy.spatial.transform import Rotation as Rot

    def max_n_data(df):
        idxs = df['pid'].values
        max_len = 0
        max_p = None
        particles = None
        frame = None
        for i in idxs:
            row = df[df['pid'] == i].reset_index()
            ns = row.loc[0, 'nb_particles_15']
            ns = eval(ns)
            if len(ns) > max_len:
                max_p = row[['frame', 'zs', 'ys', 'x_s']].values
                particles = ns
                frame = row.loc[0, 'frame']
        rows = []
        for p in particles:
            row = df[(df['frame'] == frame) & (df['particle'] == p)]
            rows.append(row)
        ndf = pd.concat(rows).reset_index(drop=True)
        return max_p, ndf
    

    def read_test_image():
        data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.zarr') # path to test image 
        img = zarr.open(data_path)
        return img


    def add_tracks(viewer, df):
        data = df[['pid', 'frame', 'zs', 'ys', 'x_s']].values
        coords = df[['x_s', 'ys', 'zs']].values
        rot = Rot.from_euler('z', 45, degrees=True) # coordinates in df are rotated -45 from original image so +45
        coords = rot.apply(coords)
        coords = coords[:, ::-1] # invert the order of the columns
        data.loc[:, ['zs', 'ys', 'x_s']] = coords
        roi_x = 88.8062132132958 * 0.504248154773493 # values found in the metadata for the test image
        roi_y = 178.823238187239 * 0.504248154773493 # roi_<axis> * px_microns
        data['x_s'] = data['x_s'] + roi_x # in the transformation the roi vals were subtracted so now add
        data['ys'] = data['ys'] + roi_y
        viewer.add_tracks(data, properties=df)
        return viewer


    def show_data_and_max_neighbours(df):
        max_p, ndf = max_n_data(df)
        image = read_test_image()
        viewer = napari.view_image(
            image, 
            scale=(2, 0.504248154773493, 0.504248154773493)) # scale to microns
            # no need to translate as the coords were comnverted back to the original
            # image in microns
        # add the tracks
        viewer = add_tracks(viewer, df)
        # plot a point with the greatest number of nearest neighbours
        viewer.add_points(max_p)
        # add the neighbour tracks
        viewer = add_tracks(viewer, ndf)


    
    #df = test_add_neighbour_lists()
    #print(df['nb_particles'].head(20))
    #data_path = os.path.join(pa_dir, 'data', 'test_data', 'test_data.parquet')
    #df.to_parquet(data_path)
    #show_data_and_max_neighbours(df)

    df = test_local_contraction()
    print(df['nb_cont_15'].max())


