import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.cluster import DBSCAN
#from sklearn import metrics
from scipy import ndimage
from tqdm import tqdm



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

