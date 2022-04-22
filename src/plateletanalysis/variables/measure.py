import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.cluster import DBSCAN
#from sklearn import metrics
from scipy import ndimage
from tqdm import tqdm
from toolz import curry



# -------------------
# Current Mesurements
# -------------------

def platelet_measurments(df):
    # add velocity
    df = finite_difference_derivatives(df)
    # add neighbour distance
    df = neighbour_distance(df)



# -------------------
# Spatial Derivatives
# ------------------- 

def finite_difference_derivatives(df):
    df = df.sort_values('pid').reset_index()
    dv_df = {
        'dvx' : np.array([np.nan, ] * len(df)).astype(np.float64),
        'dvy' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        'dvz' : np.array([np.nan, ] * len(df)).astype(np.float64),  
        'pid' : df.index.values
    }
    files = pd.unique(df['path'])
    for f in files:
        img_df = df[df['path'] == f]
        platelets = pd.unique(img_df['particle'])
        n_iter = len(platelets)
        with tqdm(total=n_iter, desc=f'Finite difference derivatives for {f}') as progress:
            for p in platelets:
                p_df = img_df[img_df['particle'] == p]
                p_df = p_df.sort_values('frame')
                idxs = p_df.index.values
                dvx = np.append(np.diff(p_df['x_s']), np.nan)
                dvy = np.append(np.diff(p_df['ys']), np.nan)
                dvz = np.append(np.diff(p_df['zs']), np.nan)
                dv_df['dvx'][idxs] = dvx
                dv_df['dvy'][idxs] = dvy
                dv_df['dvz'][idxs] = dvz
                progress.update(1)
    dv_df = pd.DataFrame(dv_df)
    dv_df.reset_index(drop=True)
    df = pd.concat([df, dv_df], axis=1)
    df['dv']=(df.dvx**2+df.dvy**2+df.dvz**2)**0.5
    df = df.drop(['pid'], axis=1)
    df['pid'] = range(len(df))
    try:
        df = df.drop(['level_0'], axis=1)
    except:
        pass
    return df


def fourier_derivatives(df):
    # Fourier transform offers a smoother way of measuring the derivative of time series measurement
    pass



# ------------------
# Neighbour Distance
# ------------------

def neighbour_distance(pc):
    t_grp=pc.set_index('pid').groupby(['path', 'frame']).apply(_nearest_neighbours)
    pc = pd.concat([pc.set_index('pid'), t_grp.set_index('pid')], axis=1)
    return t_grp


def _nearest_neighbours(pc):
    nb_count=3
    key_dist={}
    key_idx={}
    #print(len(pc))
    p1i=pc.reset_index().pid
    if len(pc)>nb_count:
        dmap=spatial.distance.squareform(spatial.distance.pdist(pc[['x_s','ys','zs']].values))
        dmap_sorted=np.sort(dmap, axis=0)
        dmap_idx_sorted=np.argsort(dmap, axis=0)
        for i in range(nb_count):
            nb_dist=(dmap_sorted[i+1,:])
            nb_idx=dmap_idx_sorted[i+1,:]
            key_dist['nb_d_' + str(i)]=nb_dist
            key_idx['nb_i_' + str(i)]=pd.Series(nb_idx).map(p1i).values.astype('int').tolist()
        key_idx.update(key_dist)
    else:
        a = np.empty((len(pc)))
        a[:] = np.nan
        key_idx['nb_i_0']=a
    df=pd.DataFrame(key_idx)
    df=pd.concat([p1i, df], axis=1)
    return df


def average_neighbour_distance(pc):
    t_grp=pc.set_index('pid').groupby(['path', 'frame']).apply(_nearest_neighbours_average)
    pc = pd.concat([pc.set_index('pid'), t_grp.set_index('pid')], axis=1).reset_index()
    return pc


def _nearest_neighbours_average(pc):
    nb_count=3
    nba_list=[5,10,15]
    key_dist={}
    key_idx={}
    #print(len(pc))
    p1i=pc.reset_index().pid
    if len(pc)>np.array(nba_list).max():
        dmap=spatial.distance.squareform(spatial.distance.pdist(pc[['x_s','ys','zs']].values))
        dmap_sorted=np.sort(dmap, axis=0)
        #dmap_idx_sorted=np.argsort(dmap, axis=0)
        for i in nba_list:
            nb_dist=(dmap_sorted[1:(i+1),:]).mean(axis=0)
            #nb_idx=dmap_idx_sorted[i+1,:]
            key_dist['nba_d_' + str(i)]=nb_dist
            #key_idx['nb_i_' + str(i)]=pd.Series(nb_idx).map(p1i).values.astype('int').tolist()
        #key_idx.update(key_dist)
    else:
        a = np.empty((len(pc)))
        a[:] = np.nan
        key_dist[('nba_d_' + str(nba_list[0]))]=a
    df=pd.DataFrame(key_dist)
    df=pd.concat([p1i, df], axis=1)
    return df


def add_neighbour_lists(df, max_dist=15):
    nb_df = {
        f'nb_particles_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'nb_disp_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'pid' : df['pid'].values
    }
    nb_df = pd.DataFrame(nb_df).set_index('pid')
    files = pd.unique(df['path'])
    for f in files:
        file_df = df[df['path'] == f]
        frames = pd.unique(df['frame'])
        for frame in frames:
            frame_wise_neigbour_lists(file_df, frame, max_dist, nb_df)
    df = pd.concat([df.set_index('pid'), nb_df], axis=1).reset_index()
    return df


def frame_wise_neigbour_lists(df, frame, max_dist, nb_df):
    f_df = df[df['frame'] == frame]
    f_df = f_df.set_index('pid') 
    idxs = f_df.index.values
    for p in idxs:
        x, y, z = f_df.loc['x_s', p], f_df.loc['ys', p], f_df.loc['zs', p]
        dx = x - f_df['x_s'].values
        dy = y - f_df['ys'].values
        dz = z - f_df['zs'].values
        disp = ((dx ** 2) + (dy ** 2) + (dz ** 2)) ** 0.5
        keep = np.where(disp < max_dist)
        nb_ps = list(f_df['particle'].values[keep])
        nb_df.loc['nb_particles', p] = nb_ps
        nb_df.loc['nb_disp', p] = list(disp[keep])


        

#TODO

def local_contraction(df):
    '''
    Local contraction is the extent to which the platelet has moved closer to its
    assigned neighbour platelets since the previous point in time

    * sum over i = 0, ..., n
    local contraction = 
        1/n * sum(
            (x_p(t) - x_i(t))**2 + 
            (y_p(t) - y_i(t))**2 + 
            (z_p(t) - x_i(t))**2 ) ** 0.5)
        - 1/n * sum(
            (x_p(t + 1) - x_i(t + 1))**2 + 
            (y_p(t + 1) - y_i(t + 1))**2 + 
            (z_p(t + 1) - x_i(t + 1))**2 ) ** 0.5)
    '''
    #TODO: finish implementing this, the computation has been worked out below
    pass


@curry
def _local_contraction(df, pid):
    # apply this to df grouped by path (i.e., single video file)
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc['nb_particles', pid]
    nbs = _ensure_list(nbs)
    nb_disp = row.loc['nb_particles', pid]
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc['frame', pid] 
    pt = row.loc['particle', pid]
    # find the location of the platelet at the next point in time
    tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
    tp1_coords = np.array([tp1_row.loc['x_s', 0], tp1_row.loc['ys', 0], tp1_row.loc['zs', 0]])
    # find the location of the current neighbours at the next point in time
    nb_coords = []
    for nb in nbs:
        nb_row = df[(df['particle'] == nb) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
        coords = np.array([nb_row.loc['x_s', 0], nb_row.loc['ys', 0], nb_row.loc['zs', 0]])
        nb_coords.append(coords)
    nb_coords = np.stack(nb_coords, axis=0)
    tp1_diff = tp1_coords - nb_coords
    tp1_disp = np.linalg.norm(tp1_diff, axis=1) # compute 2-norm for each row vector
    local_cont = np.mean(nb_disp) - np.mean(tp1_disp)
    return local_cont




def local_density(df, r=15, z_max=66):
    '''
    Density of the neighbour platelets in the sphere (radius r) around the platelet. 
    The area of the sphere is corrected for the top and the bottom of the image 

    Define the height h above or below the image:
    h = r - z             if z <= r
    h = 0                 if z < z + r < 66 
    h = z + r - 66        if z + r > 66
    where z is the height of the platelet in z (um)

    A = (4/3)πr**3 - (1/3)πh**2(3r -h)
    neighbour density = n/A
    '''
    get_density = _local_density(df, r, z_max)
    densities = df['pid'].apply(get_density)
    df[f'nb_density_{r}'] = densities
    return df



@curry
def _local_density(df, r, z_max, pid):
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc['nb_particles', pid]
    nbs = _ensure_list(nbs)
    z = row.loc['zs', pid]
    if z <= r:
        h = r - z
    elif (z + r) < z_max and (z - r) > 0:
        h = 0
    elif (z + r) > z_max:
        h = z + r - z_max
    area = ((4 * np.pi * r**3) / 3) - ((np.pi * h**2) * (3 * r - h) / 3)
    density = len(nbs) / area
    return density


def _ensure_list(list_or_string):
    if isinstance(list_or_string, str):
        nbs = eval(list_or_string)
    assert isinstance(list_or_string, list)
    return list_or_string


# ---------
# Embolysis
# ---------


def embolysis(df, r_emb=10):
    emb_df = {
        'emb_nbs' : [None, ] * len(df), 
        'pid' : df['pid'].values 
    }
    emb_df = pd.DataFrame(emb_df).set_index('pid')
    df['terminates'] = df['tracknr'] == df['nrtracks']
    term_df = df[df['terminates'] == True]
    files = pd.unique(df['path'])
    get_embolysis_freinds = _embolysis_partners(df, r_emb)
    for f in files:
        p_df = term_df[term_df['path'] == f]
        idxs = p_df['pid'].values # not index for p_df but for emb_df
        emb_ptns = p_df['pid'].apply(get_embolysis_freinds)
        emb_df.loc['emb_nbs', idxs] = emb_ptns
    df = pd.concat([df.set_index('pid'), emb_df], axis=1).reset_index()  
    return df


@curry
def _embolysis_partners(df, r_emb, pid):
    row = df[df['pid'] == pid].reset_index()
    # are any neighbours also terminating at this time point or in the next?
    nbs = row.loc['nb_particles', 0]
    nbs = _ensure_list(nbs)
    nb_disp = row.loc['nb_disp', 0]
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc['frame', 0]
    nb_ptns = []
    for i, n in enumerate(nbs):
        if nb_disp[i] <= r_emb:
            nb_row = df[df['particle'] == n].reset_index()
            if nb_row.loc['terminating', 0]:
                nb_ptns.append(n)
            else: # only need to move to next tp if not terminating
                nb_row = df[(df['particle'] == n) & (df['frame'] == frame + 1)]
                if nb_row.loc['terminating', 0]:
                    nb_ptns.append(n)
    if len(nb_ptns) == 0:
        nb_ptns = None
    return nb_ptns



# ------
# DBSCAN
# ------

def DBSCAN_cluster_1(pc):
    min_samples=5
    eps_list=[5,7.5,10,15,20]
    cl_=[]
    X=pc[['x_s','ys','zs']].values # 3D
    for eps in eps_list:
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        cl_.append(pd.DataFrame({('cl_idx_' + str(eps) ) : (labels)}))
    cl_.append(pc.reset_index().pid)
    return (pd.concat(cl_, axis=1))


def DBSCAN_cluster_2(pc):
    min_samples=5
    eps_list=np.arange(3, 31, 2)
    cl_=[]
    X=pc[['x_s','ys','zs']].values # 3D
    for eps in eps_list:
        # Compute DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        labels[labels>-1]=eps
        labels[labels==-1]=42
        cl_.append(pd.DataFrame({('cl_idx_' + str(eps) ) : (labels)}))
    cld_grp=pd.concat(cl_, axis=1)
    cld_grp=pd.DataFrame({('cld' ) : (cld_grp.min(axis=1))})
    cld_grp['pid']=(pc.reset_index().pid)
    return (cld_grp)



# ---------
# Stability
# ---------

def stability(pc):
    t_grp = pc.groupby(['path']).apply(do_tstab)
    pc = pd.concat([pc.set_index('pid'), t_grp.set_index('pid')], axis=1).reset_index()
    return pc


def do_tstab(tgrp):
    ocp_=[]
    first=True
    for i, grp in tgrp.groupby(['frame']):
        pos=grp[['x_s','ys','zs']].values
        if first:
            first=False
        else:
            dmap=spatial.distance.cdist(t1, pos)
            dmap_sorted=np.sort(dmap, axis=1)
            data=dmap_sorted[:,0]
            ocp_.append(pd.DataFrame({'stab' : data}))#, 'pid': grp.pid}))
        t1=pos
    data=np.zeros(len(pos))
    data[:]=np.nan
    ocp_.append(pd.DataFrame({'stab' : data}))#, 'pid':grp.pid}))
    ocp=pd.concat(ocp_, axis=0).reset_index()
    ocp['pid']=tgrp.reset_index().pid
    return ocp



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



