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


def points_VTK(df, path):
    bad_cols = list(df.select_dtypes('object').columns.values)
    bad_cols.append('size')
    data_columns = [col for col in df.columns.values if col not in bad_cols]
    df = df[df.nrtracks>1]
    frame_max = df.frame.max()
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
        file_path = os.path.join(path, filename)
        #data = {'v' : vtk_compat(values), 'dvx' : vtk_compat(dvx), 'dvy' : vtk_compat(dvy), 'dvz' : vtk_compat(dvz),}
        data['vector'] = (vtk_compat(dvx), vtk_compat(dvy), vtk_compat(dvz),)
        pointsToVTK(file_path, vtk_compat(x), vtk_compat(y), vtk_compat(z), data)
        print(frame, '-', end='')


# Use sml_df  = df[df['path'] == '200527_IVMTR73_Inj4_saline_exp3']
        

# -----------
# KDE Volumes
# -----------


def get_canvas_volume(df_do, micron_per_voxel=1.0): #, t_point, t_extra=2):
    #available_frames = df_do.frame.unique()
    #t_min = max(available_frames.min(), t_point-t_extra)
    #t_max = min(available_frames.max(), t_point+t_extra)
    voxels_per_micron = 1/micron_per_voxel
    extra_xyz = 10 #microns
    lim_mins = np.round(df_do[['x_s', 'ys', 'zs']].min().values) - extra_xyz
    lim_maxs = np.round(df_do[['x_s', 'ys', 'zs']].max().values) + extra_xyz
    vsize = ((lim_maxs-lim_mins)*voxels_per_micron).astype('int').tolist()
    #t_side_window = 
    #vt = np.zeros(vsize + [t_max-t_min+1], dtype = 'float32')
    return vsize, lim_mins



def get_point_volumes(df_do, vsize, lim_mins, t_point, func='gaus', t_extra=2, use_ones = True, column=None, micron_per_voxel=1.0, gamma = (5,5,5,5) ):
    voxels_per_micron = 1/micron_per_voxel
    available_frames = df_do.frame.unique()
    t_min = t_point - t_extra
    t_max = t_point + t_extra
    t_size = 2*t_extra + 1
    vt = np.zeros(vsize + [t_size], dtype = 'float32')
    for i, idf in df_do.iterrows():
        x = int(np.round(voxels_per_micron*(idf['x_s'] - lim_mins[0])))
        y = int(np.round(voxels_per_micron*(idf['ys'] - lim_mins[1])))
        z = int(np.round(voxels_per_micron*(idf['zs'] - lim_mins[2])))
        t = int(idf['frame'])
        if use_ones:
            if idf[column]:
                vt[x,y,z,(t-t_min)] += 1
        else:
            vt[x,y,z,(t-t_min)] += idf[column]*1000     
    if func == 'gaus':
        vt = ndimage.gaussian_filter(vt, gamma)
    if func == 'dist':
        maxdist = 20
        vt = ndimage.morphology.distance_transform_edt(1-vt)
        vt = maxdist - vt
    v = vt[...,t_extra] 
    return v#, lim_mins   



def export_vol(file, v, origin = (0.0, 0.0, 0.0), pixels_per_micron = 1.0, label='data'):
    data = {label : v,}
    spacing = tuple([pixels_per_micron]*3)
    origin = tuple(origin)
    #pointsToVTK('vtk_points', x, y, z, data)
    imageToVTK(file , origin=origin, spacing=spacing, cellData=None, pointData=data)


def data_colum_volume_percentile(df_sel, path, exp_tag, columns, low_perc, high_perc):
    v_shape, origin = get_canvas_volume(df_sel)
    t_extra = 2

    print(exp_tag)
    for col in columns:
        col_low = col + '_low'
        col_high = col + '_high'

        frame_max = df_sel.frame.max()
        df_do_org = df_sel[~df_sel[col].isnull()]
        for t_point in range(0, 50):#, frame_max):
            df_do = df_do_org.copy()
            #df_do[col] = df_do[col]*(100/df_do[col].mean())
            f_min = t_point-t_extra
            f_max = t_point+t_extra
            df_do = df_do[df_do.frame>=f_min]
            df_do = df_do[df_do.frame<=f_max]
            df_do[col_low] = df_do[col] <= np.percentile(df_do[col], low_perc)
            df_do[col_high] = df_do[col] > np.percentile(df_do[col], high_perc)
            #df_do['do_all'] = True
            #print(df_do.shape)

            # LOW
            vl = get_point_volumes(df_do, v_shape, origin, t_point, func='gaus', t_extra=t_extra, use_ones=True, column=col_low, micron_per_voxel=1.0, gamma = (5,5,5,5))
            export_file = f'vol_{col_low}_{t_point}'
            export_file = os.path.join(path, export_file)
            data = vtk_compat(vl)
            #export_vol(export_file, data, origin)
            #time_series_export['files'].append(dict(name = export_file, time = float(t_point)))
            print('Exported frame:', t_point, col_low, end='-')

            # HIGH
            vh = get_point_volumes(df_do, v_shape, origin, t_point, func='gaus', t_extra=t_extra, use_ones=True, column=col_high, micron_per_voxel=1.0, gamma = (5,5,5,5))
            export_file = f'vol_{col_high}_{t_point}'
            export_file = os.path.join(path, export_file)
            data = vtk_compat(vh)
            #export_vol(export_file, data, origin)
            #time_series_export['files'].append(dict(name = export_file, time = float(t_point)))
            print(col_high, end='-')

            # DIFF
            v_diff = vh-vl
            export_file = f'vol_{col}_diff_{t_point}'
            export_file = os.path.join(path, export_file)
            data = vtk_compat(v_diff)
            export_vol(export_file, data, origin)
            #time_series_export['files'].append(dict(name = export_file, time = float(t_point)))
            print('diff', end='-')

            # TOT
            v_tot = vh + vl
            export_file = f'vol_{col}_tot_{t_point}'
            export_file = os.path.join(path, export_file)
            data = vtk_compat(v_tot)
            #export_vol(export_file, data, origin)
            #time_series_export['files'].append(dict(name = export_file, time = float(t_point)))
            print('tot')   


if __name__ == '__main__':
    MIPS = '210520_IVMTR108_Inj3_MIPS_exp3'
    DMSO = '210511_IVMTR105_Inj5_DMSO2_exp3'
    d = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes'
    file_names = ('211206_mips_df.parquet', '211206_veh-mips_df.parquet')
    file_paths = [os.path.join(d, n) for n in file_names]
    mdf = pd.read_parquet(file_paths[0])
    mdf = mdf[mdf['path'] == MIPS]
    ddf = pd.read_parquet(file_paths[1])
    ddf = ddf[ddf['path'] == DMSO]
    #sp0 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/paraview/MIPS'
    #points_VTK(mdf, sp0)
    sp1 = '/Users/abigailmcgovern/Data/platelet-analysis/MIPS/paraview/DMSO'
    points_VTK(ddf, sp1)
