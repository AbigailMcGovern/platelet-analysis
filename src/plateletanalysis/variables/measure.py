import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.cluster import DBSCAN
#from sklearn import metrics
from scipy import ndimage
from tqdm import tqdm
from toolz import curry
from scipy.stats import percentileofscore, scoreatpercentile
from scipy.spatial import cKDTree
import zarr
from pathlib import Path
import os
#from plateletanalysis.variables.transform import revert_to_pixel_coords



# -------------------
# Derivative Estimates
# ------------------- 

def finite_difference_derivatives(
    df, 
    coords=('x_s', 'ys', 'zs'), # can use spherical coordinates here
    names=('dvx', 'dvy', 'dvz', 'dv'), # can name for spherical coordinates
    sample_col='path', 
    ):
    '''
    Finite difference estimates of the velocity in 3D. 

    Parameters
    ----------
    df: pandas.DataFrame
        The tracks data
    coords: 3-tuple of str
        The names of the columns containing the 3D coordinates
    names: 4-tuple of str
        The names of the columns into which to put the coordinate 
        partial derrivates (indices 0-2) and the 2-norm (index 3)
    '''
    df = df.sort_values('pid').reset_index()
    dv_df = {
        names[0] : np.array([np.nan, ] * len(df)).astype(np.float64),
        names[1] : np.array([np.nan, ] * len(df)).astype(np.float64), 
        names[2] : np.array([np.nan, ] * len(df)).astype(np.float64),  
        'pid' : df.index.values
    }
    files = pd.unique(df[sample_col])
    for f in files:
        img_df = df[df[sample_col] == f]
        platelets = pd.unique(img_df['particle'])
        n_iter = len(platelets)
        with tqdm(total=n_iter, desc=f'Finite difference derivatives for {f}') as progress:
            for p in platelets:
                p_df = img_df[img_df['particle'] == p]
                p_df = p_df.sort_values('frame')
                idxs = p_df.index.values
                dvx = np.append(np.diff(p_df[coords[0]]), np.nan)
                dvy = np.append(np.diff(p_df[coords[1]]), np.nan)
                dvz = np.append(np.diff(p_df[coords[2]]), np.nan)
                dv_df[names[0]][idxs] = dvx
                dv_df[names[1]][idxs] = dvy
                dv_df[names[2]][idxs] = dvz
                progress.update(1)
    dv_df = pd.DataFrame(dv_df)
    dv_df.reset_index(drop=True)
    df = pd.concat([df, dv_df], axis=1)
    df[names[3]]=(df.dvx**2+df.dvy**2+df.dvz**2)**0.5
    df = df.drop(['pid'], axis=1)
    df['pid'] = range(len(df))
    try:
        df = df.drop(['level_0'], axis=1)
    except:
        pass
    return df



def add_finite_diff_derivative(df, col, sample_col):
    files = pd.unique(df[sample_col])
    n_iter = 0
    for f in files:
        f_df = df[df[sample_col] == f]
        platelets = pd.unique(df['particle'])
        n_iter += len(platelets)
    #df = df.set_index('pid')
    with tqdm(total=n_iter, desc=f'Adding finite difference derivatives for {col}') as progress:
        for f in files:
            file_df = df[df[sample_col] == f]
            platelets = pd.unique(df['particle'])
            for p in platelets:
                p_df = file_df[file_df['particle'] == p]
                p_df = p_df.sort_values('frame')
                idxs = p_df.index.values
                diff = np.diff(p_df[col].values)
                diff = np.concatenate([diff, np.array([np.NaN, ])])
                df.loc[idxs, f'{col}_diff'] = diff
                progress.update(1)
    df = df.reset_index(drop=True)
    return df


def fourier_derivatives(df):
    # Fourier transform offers a smoother way of measuring the derivative of time series measurement
    pass


# -----------
# Point Depth
# -----------


def point_depth(pc):
    t_grp = pc.groupby(['path']).apply(point_depth_2).reset_index() 
    t_grp = t_grp.drop(['path'], axis=1)
    pc = pd.concat([pc.set_index('pid'), t_grp.set_index('pid')], axis=1).reset_index()  
    try:
        pc = pc.drop(['level_1'], axis=1)
    except:
        pass
    return pc


def point_depth_1(pos):
    fill_dist=15
    pos_max=pos[['x_s', 'ys', 'zf']].max()
    zsize=int(pos_max['zf']*1.5)
    xsize=int(pos_max['x_s']*1.1)
    ysize=int(pos_max['ys']*1.1)
    pcc=np.zeros((xsize+2,ysize+2,zsize+2))
    pc_pos=pos[['x_s', 'ys', 'zf']].values
    zfloor=1
    pc_pos[pc_pos<0]=0
    pc_pos=pc_pos.astype('int').T
    pc_pos=pc_pos.tolist()
    pcc[pc_pos]=1
    pcc[:,:,:zfloor]=1
    pcd=ndimage.morphology.distance_transform_edt(pcc==0)
    pca=pcd<fill_dist
    pca=ndimage.binary_fill_holes(pca)
    pcad=ndimage.morphology.distance_transform_edt(pca)
    pdepth=pcad[pc_pos]-fill_dist
    depth_grp=pd.DataFrame({'depth' : (pdepth)})
    depth_grp['pid']=pos.reset_index().pid
    return depth_grp


def point_depth_2(pos_grp):
    fill_dist=15
    pos_max=pos_grp[['x_s', 'ys', 'zf']].max()
    zsize=int(pos_max['zf']*1.5)
    xsize=int(pos_max['x_s']*1.1)
    ysize=int(pos_max['ys']*1.1)
    depth_grp_=[]
    for i, pos in pos_grp.groupby('frame'):
        pcc=np.zeros((xsize+2,ysize+2,zsize+2))
        pc_pos=pos[['x_s', 'ys', 'zf']].values
        zfloor=1
        pc_pos[pc_pos<0]=0
        pc_pos=pc_pos.astype('int').T
        pc_pos=pc_pos.tolist()
        pcc[pc_pos]=1
        pcc[:,:,:zfloor]=1
        pcd=ndimage.morphology.distance_transform_edt(pcc==0)
        pca=pcd<fill_dist
        pca=ndimage.binary_fill_holes(pca)
        pcad=ndimage.morphology.distance_transform_edt(pca)
        pdepth=pcad[pc_pos]-fill_dist
        depth_grp=pd.DataFrame({'depth' : (pdepth)})
        depth_grp['pid']=pos.reset_index().pid
        depth_grp_.append(depth_grp)
    depth_grp_=pd.concat(depth_grp_, axis=0)
    return depth_grp_



# -------
# Entropy
# -------

def length_entropy(df, track_no_frames=None, particle_id='particle'):
    '''
    Very crude estimate of the amount of information provided by the
    length of a platelet track (i.e., platelets which are tracked for
    longer provide more information as they make up only a small proportion
    of all tracked platelets)

    Notes
    -----
    - P(len) is cestimated as proportion of tracks with a given length
    - entropy is caclulated in bits according to 
        h(xi) = - P(xi) • log2(P(xi))
    '''
    if track_no_frames is None:
        assert particle_id is not None
        pass # apply function to add this 
    else:
        unique_ps = np.unique(df[particle_id].values)
        lens = []
        for p in unique_ps:
            lil_df = df[df[particle_id] == p]
            lens.append(lil_df.loc[0, track_no_frames]) # lens for each platelet
        unique_lens = np.unique(lens)
        p_len = []
        n_total = len(lens) # number of uniquely identified platelets
        for l in unique_lens:
            # get the 'probability' of a track having a certain length
            n_match = len(np.where(lens == l)[0])
            p = n_match / n_total
            p_len.append(p)
        h_len = []
        for p in p_len:
            h = - p * np.log2(p)
            h_len.append(h)
        entropy_df = {
            particle_id : unique_ps, 
            track_no_frames : lens, 
            'probability' : p, 
            'entropy_bits' : h
        }
        entropy_df = pd.DataFrame(entropy_df)
        return entropy_df


# -----------
# Contraction
# -----------

def contractile_motion(pc):
    cont_grp = pc.reset_index().groupby(['path']).apply(contract)
    cont_grp = cont_grp.set_index('pid').sort_index()
    cont_grp = cont_grp.reset_index()
    pc = pc.reset_index()
    pc = pd.concat([pc.set_index('pid'), cont_grp.set_index('pid')], axis=1).reset_index()
    try:
        pc = pc.drop(['level_0'], axis=1)
    except:
        pass
    return pc


def contract(t2):
    t2['cont']=((-t2['x_s'])*t2['dvx'] + (-t2['ys'])*t2['dvy'] + (-t2['zf'])*t2['dvz'] )/((t2['x_s'])**2 + (t2['ys'])**2 + (t2['zf'])**2)**0.5
    t2['cont_p']=t2.cont/t2.dv
    return pd.DataFrame({'cont' : (t2['cont']), 'cont_p' : (t2['cont_p']), 'pid':t2['pid']})


# -----------
# Path Length
# -----------

def path_disp_n_tortuosity(df):
    '''
    Add the variables path length (path_len), displacement (disp), 
    '''
    files = pd.unique(df['path'])
    for f in files:
        img_df = df[df['path'] == f]
        platelets = pd.unique(img_df['particle'])
        n_iter = len(platelets)
        with tqdm(total=n_iter, desc=f'Path length, displacement, and, tortuosity for {f}') as progress:
            for p in platelets:
                p_df = img_df[img_df['particle'] == p].sort_values(by='frame')
                idxs = p_df.index.values
                pathlen = np.cumsum(np.abs(p_df['dv'].values))
                df.loc[idxs, 'path_len'] = pathlen
                t0 = p_df['frame'].min()
                t0_i = p_df[p_df['frame'] == t0].index.values[0]
                x0, y0, z0 = p_df.loc[t0_i, 'x_s'], p_df.loc[t0_i, 'ys'], p_df.loc[t0_i, 'zs']
                x_disp = p_df['x_s'] - x0
                y_disp = p_df['ys'] - y0
                z_disp = p_df['zs'] - z0
                disp = ((x_disp ** 2) + (y_disp ** 2) + (z_disp ** 2)) ** 0.5
                df.loc[idxs, 'disp'] = disp
                df.loc[idxs, 'tort'] = pathlen / disp
                progress.update(1)
    return df



def platelet_displacement(df):
    df = df.sort_values('pid').reset_index()
    d_df = {
        'disp' : np.array([np.nan, ] * len(df)).astype(np.float64),
        'disp_sum' : np.array([np.nan, ] * len(df)).astype(np.float64),
        'tort' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        'tort_p' : np.array([np.nan, ] * len(df)).astype(np.float64)
    }
    files = pd.unique(df['path'])
    for f in files:
        img_df = df[df['path'] == f]
        platelets = pd.unique(img_df['particle'])
        n_iter = len(platelets)
        with tqdm(total=n_iter, desc=f'Finite difference derivatives for {f}') as progress:
            for p in platelets:
                p_displacement_from_tmin1(img_df, p, d_df)
    d_df = pd.DataFrame(d_df)
    d_df.reset_index(drop=True)
    df = pd.concat([df, d_df], axis=1)
    df = df.drop(['pid'], axis=1)
    df['pid'] = range(len(df))
    try:
        df = df.drop(['level_0'], axis=1)
    except:
        pass
    return df


def p_displacement_from_tmin1(df, particle, d_df):
    p_df = df[df['particle'] == particle]
    p_df = p_df.sort_values('frame')
    idxs = p_df.index.values
    dx = np.append(np.diff(0., p_df['x_s']))
    dy = np.append(np.diff(0., p_df['ys']))
    dz = np.append(np.diff(0., p_df['zs']))
    disp = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
    disp_sum = np.cumsum(disp)
    tort = disp_sum / d_df['dist_c'].values
    tort_p = disp_sum[-1] / d_df['dist_c'].values[-1]
    tort_p = tort_p * len(disp)
    d_df.loc['disp', idxs] = disp
    d_df.loc['disp_sum', idxs] = disp_sum
    d_df.loc['tort', idxs] = tort
    d_df.loc['tort_p', idxs] = tort_p



# -------------
# thromus percentile
# -------------

def find_clot_percentile_structure(
        df, 
        u_percentiles=(99, 95, 90, 80, 70, 60), 
        l_percentiles=(1, 5, 10, 20, 30, 40) 
    ):
    '''
    for each clot, find x-y coordinates of each 
    '''
    files = pd.unique(df['path'])
    percentiles = 100 - (np.array(l_percentiles) * 2)
    angles = np.linspace(0, np.pi, 100)
    results = {
        'path' : [],
        #'frame' : [],
        'zs' : [], 
        'ys' : [], 
        'x_s' : [], 
        'rotation' : [], 
        'percentile' : []
    }
    frames = pd.unique(df['frame'])
    #z_levels = pd.unique(df['zs'])
    z_levels = np.linspace(0, 64, 10)
    n_iter = len(files) * len(z_levels) * len(angles) # * len(frames)
    with tqdm(total=n_iter) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            #frames = pd.unique(fdf['frame'])
            #for t in frames:
                #tdf = fdf[fdf['frame'] == t]
            z_max = fdf['zs'].max()
            z_levels = np.linspace(0, z_max, 10)
            for i in range(len(z_levels) - 1):
                z = (z_levels[i] + z_levels[i + 1]) / 2
                zdf = fdf[(fdf['zs'] >= z_levels[i]) & (fdf['zs'] <= z_levels[i + 1])]
                zdf = zdf.sort_values(by='x_s')
                xs = zdf['x_s'].values
                ys = zdf['ys'].values
                if len(xs > 0):
                    for rot in angles:
                        new_xs, new_ys = rotate(xs, ys, rot)
                        lower = np.percentile(new_xs, l_percentiles)
                        upper = np.percentile(new_xs, u_percentiles)
                        # need to figure out how to transform back into original xy
                        # x = r cos theta « where r is the val for the percentile in rotated space
                        xl = lower * np.cos(rot)
                        xu = upper * np.cos(rot)
                        # y = r sin theta « and theta is the rotation angle in rad
                        yl = lower * np.sin(rot)
                        yu = upper * np.sin(rot)
                        # add the data
                        for i in range(len(xu)):
                            results['path'].append(f)
                            #results['frame'] = t
                            results['zs'].append(z)
                            results['ys'].append(yu[i])
                            results['x_s'].append(xu[i])
                            results['rotation'].append(rot)
                            results['percentile'].append(percentiles[i])
                        for i in range(len(xl)):
                            results['path'].append(f)
                            #results['frame'] = t
                            results['zs'].append(z)
                            results['ys'].append(yl[i])
                            results['x_s'].append(xl[i])
                            results['rotation'].append(rot + np.pi)
                            results['percentile'].append(percentiles[i])
                progress.update(1)
    results = pd.DataFrame(results)
    return results


def rotate(x, y, rot):
    xs = x * np.cos(rot) - y * np.sin(rot)
    ys = x * np.sin(rot) + y * np.cos(rot)  
    return xs, ys


def find_density_percentiles_img(
        df, 
        k=3, 
        eps=2, 
        save=None,
    ):
    files = pd.unique(df['path'])
    images = []
    t = df['frame'].max()
    z = np.round(df['zs'].max()).astype(int)
    y_min = df['ys'].min()
    y = np.round(df['ys'].max() - y_min).astype(int)
    x_min = df['x_s'].min()
    x = np.round(df['x_s'].max()- x_min).astype(int)
    with tqdm(total=len(files) * t) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            idxs = fdf.index.values
            densities = fdf['nb_density_15']
            percentiles = np.array([percentileofscore(densities, d) for d in densities])
            df.loc[idxs, 'nd15_percentile'] = percentiles
            fdf = df[df['path'] == f]
            img = np.zeros((t, z, y, x))
            for t_ in range(t):
                tdf = fdf[fdf['frame'] == t_]
                idxs = tdf.index.values
                coords = tdf[['zs', 'ys', 'x_s']].values
                coords = cKDTree(coords) # cKDTree
                for z_ in range(z):
                    for y_ in range(y):
                        for x_ in range(x):
                            # interpolate the density percentile by finding the nearest platelets (prob CKD tree)
                            yy = y_ + y_min
                            xx = x_ + x_min
                            dd, ii = coords.query([z_, yy, xx], k=k, eps=eps)
                            idx = idxs[ii]
                            dd = dd / np.sum(dd) 
                            dd = 1 - dd # get the weight of the neighbour
                            nb_pct = tdf.loc[idx, 'nd15_percentile'].values
                            percentile = np.sum(dd * nb_pct)
                            if percentile > 2:
                                img[t_, z_, y_, x_] = percentile
            progress.update(1)
        images.append(img)
    images = np.stack(images)
    if save is not None:
        sn = Path(save).stem
        sdir = Path(save).parents[0]
        save_order = os.path.join(sdir, sn)
        zarr.save(save)
        with open(save_order, 'w') as f:
            f.write(str(files))
    return df, images
    


def find_density_percentiles(
        df, 
        k=3, 
        eps=2, 
        save=None,
    ):
    files = pd.unique(df['path'])
    with tqdm(total=len(files)) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            idxs = fdf.index.values
            densities = fdf['nb_density_15']
            percentiles = np.array([percentileofscore(densities, d) for d in densities])
            df.loc[idxs, 'nd15_percentile'] = percentiles
            progress.update(1)
    return df


# ---------------
# Fibrin distance
# ---------------

# in the platelet segmentation directory there is code for segmenting fibrin and generating a distance transform
# for each pixel in the volume. Distance is in microns. 

def assign_fibrin_distance(df, dt_dict):
    for f in dt_dict.keys():
        fdf = df[df['path'] == f]
        idxs = (
            np.round(fdf['frame'].values).astype(int), 
            np.round(fdf['z_pixels'].values).astype(int), 
            np.round(fdf['y_pixels'].values).astype(int), 
            np.round(fdf['x_pixels'].values).astype(int)
            )
        dt_img = dt_dict[f]
        dt_img = np.array(dt_img)
        dist = dt_img[idxs]
        df_idxs = fdf.index.values
        df.loc[df_idxs, 'fibrin_dist'] = dist
    return df


def fibrin_cotraction(df):
    df = add_finite_diff_derivative(df, 'fibrin_dist')
    df['fibrin_cont'] = - df['fibrin_dist_diff'].values
    df = df.drop(columns=['fibrin_dist_diff'])
    return df
    

# -----------------------
# Quantile normailisation
# -----------------------

def quantile_normalise_variables(
        df, 
        vars=('phi', 'phi_diff', 'rho', 'rho_diff', 'theta', 'theta_diff', 'zs')
    ):
    files = pd.unique(df['path'])
    with tqdm(total=len(files)*len(vars)) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            for v in vars:
                vn = v + '_pcnt'
                idxs = fdf.index.values
                values = fdf[v].values
                percentiles = np.array([percentileofscore(values, d) for d in values])
                df.loc[idxs, vn] = percentiles
                progress.update(1)
    return df



def quantile_normalise_variables_frame(
        df, 
        vars=('nb_density_15', )
    ):
    files = pd.unique(df['path'])
    t_max = int(df['frame'].max())
    with tqdm(total=len(files)*len(vars)*t_max) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            for t in range(t_max):
                t_df = fdf[fdf['frame'] == t]
                for v in vars:
                    vn = v + '_pcntf'
                    idxs = t_df.index.values
                    values = t_df[v].values
                    percentiles = np.array([percentileofscore(values, d) for d in values])
                    df.loc[idxs, vn] = percentiles
                    progress.update(1)
    return df



# ---------------
# Smooth variable
# ---------------

def smooth_variables(
    df, 
    vars=('phi_diff', 'rho_diff', 'theta_diff', 'dv')
    ):
    files = pd.unique(df['path'])
    with tqdm(total=len(files)) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            ps = pd.unique(fdf['particle'])
            for p in ps:
                pdf = fdf[fdf['particle'] == p]
                pdf = pdf.sort_values(by=['frame'])
                idxs = pdf.index.values
                for v in vars:
                    rolling = pdf[v].rolling(5).mean()
                    vn = v + '_rolling'
                    df.loc[idxs, vn] = rolling
            progress.update(1)
    return df



# ---------
# Stability
# ---------
# found in 1(3)_Load data&analysis.ipynb

def stability(df):
    t_grp = df.groupby('path').apply(do_tstab)
    stab_grp = t_grp.reset_index()[['stab', 'pid']]
    stab_grp = stab_grp.set_index('pid').sort_index()
    df = df.merge(stab_grp, on = ['pid'])
    df = df.loc[:,~df.columns.duplicated()].copy()
    return df


def do_tstab(tgrp):
    ocp_ = []
    first = True
    #print(len(tgrp))
    for i, grp in tgrp.groupby(['frame']):
        pos  =grp[['x_s','ys','zs']].values
        if first:
            first = False
        else:
            dmap = spatial.distance.cdist(t1, pos)
            dmap_sorted = np.sort(dmap, axis=1)
            data = dmap_sorted[:,0]
            ocp_.append(pd.DataFrame({'stab' : data}))#, 'pid': grp.pid}))
        t1 = pos
    data = np.zeros(len(pos))
    data[:] = np.nan
    ocp_.append(pd.DataFrame({'stab' : data}))#, 'pid':grp.pid}))
    ocp = pd.concat(ocp_, axis=0).reset_index()
    ocp['pid'] = tgrp.reset_index().pid
    return ocp


# --------------------
# Contractile movement
# --------------------


def contraction(df):
    cont_grp = df.groupby(['path']).apply(contract)
    cont_grp = cont_grp.set_index('pid').sort_index()
    df = df.merge(cont_grp, on = ['pid'])
    return df


def contract(t2):
    t2['cont'] = ((-t2['x_s'])*t2['dvx'] + (-t2['ys'])*t2['dvy'] + (-t2['zf'])*t2['dvz'] )/((t2['x_s'])**2 + (t2['ys'])**2 + (t2['zf'])**2)**0.5
    t2['cont_p'] = t2.cont/t2.dv
    return pd.DataFrame({'cont' : (t2['cont']), 'cont_p' : (t2['cont_p']), 'pid':t2['pid']})
