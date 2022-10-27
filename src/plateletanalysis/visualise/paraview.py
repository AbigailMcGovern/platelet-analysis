import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import ndimage
from pyevtk.hl import imageToVTK, gridToVTK, pointsToVTK, polyLinesToVTK
from nd2reader import ND2Reader
import json
import pyevtk
from scipy import ndimage
import os
from scipy.spatial.transform import Rotation as Rot
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor



def vtk_compat(m):
    return np.ascontiguousarray(m.astype('float32'))

def vtk_compat(m):
    return np.ascontiguousarray(m.astype('float32').squeeze())

def make_path(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_files(path, pattern):
    p = Path(path).rglob(pattern)
    files = sorted([str(x).replace('\\', '/') for x in p if (x.is_file() & (x.stat().st_size!=0))])
    return files


# ------
# Points
# ------


def points_for_paraview(df, inj_name, save_dir, save_name):
    #data_columns = ['frame', 'c0_mean', 
     #          'c1_mean', 'ca_corr', 'dvx', 'dvy', 'dvz', 'dv',
      #         'particle', 'cont','nrtracks','tracknr',
       #        'nd15_percentile', 'nb_ca_corr_15', 'fibrin_dist_pcnt', 'nb_density_15_pcntf']
    df = df[df['path'] == inj_name]
    columns = list(df.columns.values)
    data_columns = []
    for col in columns:
        if not isinstance(df[col].values[0], str):
            data_columns.append(col)
    df = df[df.nrtracks>10]
    frame_max = df.frame.max()
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    for frame in range(frame_max):
        df_do = df[df.frame==frame]
        data = {}
        for col in data_columns:
            data[col] = vtk_compat(df_do[[col]].values)

        x = df_do[['x_s']].values
        y = df_do[['ys']].values
        z = df_do[['zs']].values

        dvx = df_do[['dvx']].values
        dvy = df_do[['dvy']].values
        dvz = df_do[['dvz']].values
        filename = f'points_{frame}'
        file_path = os.path.join(save_path, filename)
        #data = {'v' : vtk_compat(values), 'dvx' : vtk_compat(dvx), 'dvy' : vtk_compat(dvy), 'dvz' : vtk_compat(dvz),}
        data['vector'] = (vtk_compat(dvx), vtk_compat(dvy), vtk_compat(dvz),)
        pointsToVTK(file_path, vtk_compat(x), vtk_compat(y), vtk_compat(z), data)
        print(frame, '-', end='')

