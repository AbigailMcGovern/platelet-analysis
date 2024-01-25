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


def points_VTK(
        df, 
        path, 
        coord_cols = ('x_s', 'ys', 'zs'),
        dv_cols = ('dvx', 'dvy', 'dvz'),
        data_columns = ('frame', 'c0_mean', 
                        'c1_mean', 'ca_corr', 
                        'dvx', 'dvy', 'dvz', 'dv',
                        'particle', 'cont','nrtracks',
                        'tracknr', 'nd15_percentile', 
                        'nb_ca_corr_15', 'fibrin_dist_pcnt', 
                        'nb_density_15_pcntf')
    ):
    df = df[df.nrtracks>10]
    frame_max = df.frame.max()
    for frame in range(frame_max):
        df_do = find_decent_frame(df, frame, frame_max)
        data = {}
        for col in data_columns:
            data[col] = vtk_compat(df_do[[col]].values)
        x = df_do[[coord_cols[0]]].values
        x = vtk_compat(x)
        y = df_do[[coord_cols[1]]].values
        y = vtk_compat(y)
        z = df_do[[coord_cols[2]]].values
        z = vtk_compat(z)
        dvx = df_do[[dv_cols[0]]].values
        dvx = vtk_compat(dvx)
        dvy = df_do[[dv_cols[1]]].values
        dvy = vtk_compat(dvy)
        dvz = df_do[[dv_cols[2]]].values
        dvz = vtk_compat(dvz)
        filename = f'points_{frame}'
        file_path = os.path.join(path, filename)
        #data = {'v' : vtk_compat(values), 'dvx' : vtk_compat(dvx), 'dvy' : vtk_compat(dvy), 'dvz' : vtk_compat(dvz),}
        data['vector'] = (dvx, dvy, dvz,)
        pointsToVTK(file_path, x, y, z, data)
        print(frame, '-', end='')


def find_decent_frame(df, frame, max_frame):
    df_do = df[df.frame==frame]
    if len(df_do) == 0 and frame >= 0 and frame < max_frame - 10:
        return find_decent_frame(df, frame + 1, max_frame)
    elif len(df_do) == 0 and frame >= max_frame - 10:
        return find_decent_frame(df, frame - 1, max_frame)
    elif len(df_do) > 0:
        return df_do


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


# -----------------
# KDE Vector Volume
# -----------------

def kde_vector_volume(df_sel, path, exp_tag, t_extra = 2):
    frame_max = df_sel.frame.max()
    columns = ['dvx', 'dvy', 'dvz']
    v_shape, origin = get_canvas_volume(df_sel)
    print(exp_tag, 'Exported frames:')
    for t_point in range(0, frame_max):
        df_do = df_sel.copy()
        f_min = t_point-t_extra
        f_max = t_point+t_extra
        df_do = df_do[df_do.frame>=f_min]
        df_do = df_do[df_do.frame<=f_max]
        vs = []
        #vt, origin = get_canvas_volume(df, t_point, t_extra=2)
        for column in columns:
            v = get_point_volumes(df_do, v_shape, origin, t_point, func='gaus', t_extra=t_extra, use_ones = False, column=column, micron_per_voxel=1.0, gamma = (5,5,5,5))
            vs.append(v)
        export_file = f'vector_vol_{t_point}'
        export_file = os.path.join(path, export_file)
        data = (vtk_compat(vs[0]),vtk_compat(vs[1]),vtk_compat(vs[2]))
        export_vol(export_file, data, origin)
        #time_series_export['files'].append(dict(name = export_file, time = float(t_point)))
        print(t_point, end='-')



if __name__ == '__main__':
    #do = 'finding_and_following'
    do = 'toroidal'
    if do == 'finding_and_following':
        tp0 = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/tracking/200910_MxV_hir_600is.parquet'
        tp1 = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_veh-sq_df.parquet'
        n1 = '200519_IVMTR69_Inj4_dmso_exp3'
        df0 = pd.read_parquet(tp0)
        #df1 = pd.read_parquet(tp1)
        #df1 = df1[df1['path'] == n1]
        df0['xs'] = df0['xs'] / 0.32 * 0.5
        df0['ys'] = df0['ys'] / 0.32 * 0.5
        save_dir_0 = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/tracking/paraview/ex-vivo-demo'
        save_dir_1 = '/Users/abigailmcgovern/Data/iterseg/invitro_platelets/ACBD/mouse/tracking/paraview/in-vivo-demo'
        data_columns = ['frame', 'nrtracks', 'nb_density_15']
        points_VTK(df0, save_dir_0, data_columns=data_columns, coord_cols=('xs', 'ys', 'zs'))
        #points_VTK(df1, save_dir_1, data_columns=data_columns)
    
    elif do == 'toroidal':
        from plateletanalysis.variables.neighbours import local_contraction
        p = '/Users/abigailmcgovern/Data/platelet-analysis/dataframes/211206_saline_df_toroidal-coords-1.parquet'
        df = pd.read_parquet(p)
        #df = pd.read_parquet('/Users/abigailmcgovern/Data/platelet-analysis/dataframes/saline_ctrl_biva_cang_par4.parquet')
        inj = '200527_IVMTR73_Inj4_saline_exp3'
        df = df[df['path'] == inj]
        df = local_contraction(df)
        data_columns = ['frame', 'nrtracks', 'nb_density_15', 'tor_theta', 'tor_theta_diff', 'nb_cont_15']
        #sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_8/'
        sd = '/Users/abigailmcgovern/Data/platelet-analysis/toroidal_paper/figure_9/local_cont_para_1'
        points_VTK(df, sd, data_columns=data_columns, coord_cols=('x_s', 'ys', 'zs'))

