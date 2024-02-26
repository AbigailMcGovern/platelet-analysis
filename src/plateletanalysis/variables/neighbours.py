from tokenize import group
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from toolz import curry
from tqdm import tqdm
from distributed import Client, as_completed
from .measure import add_finite_diff_derivative
from scipy import spatial


# ------------------
# Finding Neighbours
# ------------------
# Feat. cKD tree


def add_neighbour_lists(df, max_dist=15, sample_col='path', coords=('x_s', 'ys', 'zs')):
    coords = list(coords)
    nb_df = {
        f'nb_particles_{max_dist}' : np.array([None, ] * len(df)).astype(np.float64), 
        f'nb_disp_{max_dist}' : np.array([np.nan, ] * len(df)).astype(np.float64), 
        f'pid' : df['pid'].values
    }
    nb_df = pd.DataFrame(nb_df).set_index('pid')
    files = pd.unique(df[sample_col])
    frames = pd.unique(df['frame'])
    n_iter = len(files) * len(frames)
    with tqdm(total=n_iter, desc='Adding neighbour lists') as progress:
        for f in files:
            file_df = df[df[sample_col] == f]
            frames = pd.unique(df['frame'])
            for frame in frames:
                frame_wise_neigbour_lists(file_df, frame, max_dist, nb_df, coords)
                progress.update(1)
    df = pd.concat([df.set_index('pid'), nb_df], axis=1).reset_index()
    return df


def frame_wise_neigbour_lists(df, frame, max_dist, nb_df, coords):
    f_df = df[df['frame'] == frame]
    f_df = f_df.set_index('pid') 
    idxs = f_df.index.values
    # get the coodinates as an array (r = points, c = dims)
    p_coords = f_df[coords].values
    tree1 = cKDTree(p_coords, copy_data=True)
    tree2 = cKDTree(p_coords, copy_data=True)
    sdm = tree1.sparse_distance_matrix(tree2, max_dist)
    array = sdm.toarray()
    for p in range(array.shape[0]):
        # get the non zero values for the platelet
        p_dists = array[p, :]
        p_idxs = np.where(p_dists > 0) # indicies to find pid
        keep = idxs[p_idxs] # pids that should be kept
        disps = list(p_dists[p_idxs])
        # add to the data frame
        pidx = idxs[p]
        nb_ps = list(f_df.loc[keep, 'particle'])
        nb_df.loc[pidx, f'nb_particles_{max_dist}'] = str(nb_ps)
        nb_df.loc[pidx, f'nb_disp_{max_dist}'] = str(disps)


# ----------------------
# Local Kinetic Measures
# ----------------------
    
#TODO

def local_contraction(df, r=15):
    """
    This measures how much closer neighbour platelets get to a platelet between
    consecutive time points. 
    """
    dist_col = f'nb_disp_{r}'
    df[dist_col] = df[dist_col].apply(eval)
    df[f'av_nb_disp_{r}'] = df[dist_col].apply(average_distance)
    df = df.sort_values('time (s)', ascending=False)
    lc_finc = dist_diffs(f'av_nb_disp_{r}')
    #df[f'nb_cont_{r}'] = df.groupby(['particle', 'path']).apply(lc_finc) # unsure why this doesn't work
    n = len(df.groupby(['particle', 'path'])[f'av_nb_disp_{r}'].mean())
    with tqdm(total=n, desc='local contraction') as progress:
        for k, grp in df.groupby(['particle', 'path']):
            idxs = grp.index.values
            vals = lc_finc(grp)
            df.loc[idxs, f'nb_cont_{r}'] = vals
            progress.update(1)
    return df


def average_distance(dlist):
    if len(dlist) > 0:
        return np.sum(dlist) / len(dlist)
    else:
        return 15.0



@curry
def dist_diffs(var, grp):
    # group by platelet ("particle")
    s = len(grp)
    diff = np.diff(grp[var].values)
    diff = np.concatenate([[np.nan, ], diff])
    return - diff * 0.32 # adjust for time to get µm/s


def local_contraction_old(df, r=15):
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
    files = pd.unique(df['path'])
    n_iter = len(df)
    client = Client()
    print(client.dashboard_link)
    dfs = [df[df['path'] == f].reset_index(drop=True) for f in files]
    #with tqdm(total=n_iter, desc='Adding neighbour contraction') as progress:
    filewise_local_contraction = _filewise_local_contraction(r)
    df = df.set_index('pid')
    big_future = client.scatter(dfs) 
    fs = client.map(filewise_local_contraction, big_future)
    for future, result in as_completed(fs, with_results=True):
        df.loc[result[0], f'nb_cont_{r}'] = result[1]
    client.close()
    df = df.reset_index()
    return df



@curry
def _filewise_local_contraction(r,  f_df):
    #f_df = df[df['path'] == f].reset_index()
    #f_df = dd.from_pandas(f_df, 1)
    get_contraction = _local_contraction(f_df, r)
    idxs = f_df['pid'].values
    conts = []
    fn = f_df.loc[0, 'path']
    #conts = f_df['pid'].map(get_contraction, meta=pd.Series({f'nb_cont_{r}' : []}, dtype=np.float64))
    with tqdm(total=len(f_df), desc=f'Local contraction: {fn}') as progress:
        for p in idxs:
            cont = get_contraction(p)
            conts.append(cont)
            progress.update(1)
        #df.loc[idxs, f'nb_cont_{r}'] = conts
        #print('file done')
    return idxs, conts


@curry
def _local_contraction(df, r, pid):
    # apply this to df grouped by path (i.e., single video file)
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc[pid, f'nb_particles_{r}']
    nbs = _ensure_list(nbs)
    nb_disp = row.loc[pid, f'nb_disp_{r}']
    nb_disp = _ensure_list(nb_disp)
    frame = row.loc[pid, 'frame'] 
    pt = row.loc[pid, 'particle']
    # find the location of the platelet at the next point in time
    if row.loc[pid, 'nrtracks'] > row.loc[pid, 'tracknr']:
        tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
        inc = 1
        if len(tp1_row) == 0:
            tp1_row = df[(df['particle'] == pt) & (df['frame'] == frame + 2)].reset_index()
            inc = 2
        if len(tp1_row) > 0:
            tp1_coords = np.array([tp1_row.loc[0, 'x_s'], tp1_row.loc[0, 'ys'], tp1_row.loc[0, 'zs']])
            # find the location of the current neighbours at the next point in time
            nb_coords = []
            for nb in nbs:
                nb_row = df[(df['particle'] == nb) & (df['frame'] == frame + 1)].reset_index() # would have to include path to generalise
                if len(nb_row) > 0:
                    coords = np.array([nb_row.loc[0, 'x_s'], nb_row.loc[0, 'ys'], nb_row.loc[0, 'zs']])
                    nb_coords.append(coords)
            if len(nb_coords) > 0:
                nb_coords = np.stack(nb_coords, axis=0)
                tp1_diff = tp1_coords - nb_coords
                tp1_disp = np.linalg.norm(tp1_diff, axis=1) # compute 2-norm for each row vector
                local_cont = np.mean(nb_disp) - (np.mean(tp1_disp)/inc)
            else: 
                local_cont = np.NaN
        else: 
            local_cont = np.NaN
    else: 
        local_cont = np.NaN
    return local_cont


# ---------------------
# Local Static Measures
# ---------------------

def local_density(df, r=15, z_max=66, sample_col='path'):
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
    df = df.set_index('pid')
    files = pd.unique(df[sample_col])
    n_iter = len(df)
    sphere_size = ((4 * np.pi * r**3) / 3)
    with tqdm(total=n_iter, desc='Adding neighbour density') as progress:
        for f in files:
            f_df = df[df[sample_col] == f].reset_index()
            idxs = f_df['pid'].values
            get_density = _local_density(f_df, r, z_max, sphere_size)
            #densities = f_df['pid'].apply(get_density)
            densities = []
            for p in idxs:
                density = get_density(p)
                densities.append(density)
                progress.update(1)
            df.loc[idxs, f'nb_density_{r}'] = densities
    df = df.reset_index()
    return df



@curry
def _local_density(df, r, z_max, sphere_size, pid):
    row = df[df['pid'] == pid].set_index('pid')
    nbs = row.loc[pid, f'nb_particles_{r}']
    nbs = _ensure_list(nbs)
    z = row.loc[pid, 'zs']
    if z <= r:
        h = r - z
        area = sphere_size - ((np.pi * h**2) * (3 * r - h) / 3)
    elif (z + r) > z_max:
        h = z + r - z_max
        area = sphere_size - ((np.pi * h**2) * (3 * r - h) / 3)
    else:
        area = sphere_size
    density = len(nbs) / area
    return density



def local_calcium(df, r=15):
    files = pd.unique(df['path'])
    df = df.set_index('pid')
    t_max = df['frame'].max()
    with tqdm(total=len(files)*t_max) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            for t in range(t_max):
                tdf = fdf[fdf['frame'] == t]
                pids = tdf.index.values
                df1 = tdf.reset_index()
                df1 = df1.set_index('particle')
                for p in pids:
                    ca_mean = get_calcium(df, df1, r, p)
                    df.loc[p, f'nb_ca_corr_{r}'] = ca_mean
                progress.update(1)
    df = df.reset_index()
    return df



def get_calcium(df, df1, r, pid):
    row = df.loc[pid, :]
    nbs = _ensure_list(row[f'nb_particles_{r}'])
    if len(nbs) > 0:
        calcium = [df1.loc[nb, 'ca_corr'] for nb in nbs]
        mean = np.mean(calcium)
    else:
        mean = 0
    return mean


def local_variable_mean(df, var, r=15):
    files = pd.unique(df['path'])
    if 'level_0' in df.columns.values:
        df = df.drop(columns=['level_0'])
    df = df.set_index('pid')
    t_max = df['frame'].max()
    with tqdm(total=len(files)*t_max) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            for t in range(t_max):
                tdf = fdf[fdf['frame'] == t]
                pids = tdf.index.values
                df1 = tdf.reset_index()
                df1 = df1.set_index('particle')
                for p in pids:
                    mean = get_variable(df, df1, r, p, var)
                    df.loc[p, f'nb_{var}_{r}'] = mean
                progress.update(1)
    df = df.reset_index()
    return df



def get_variable(df, df1, r, pid, var):
    row = df.loc[pid, :]
    nbs = row[f'nb_particles_{r}']
    nbs = _ensure_list(nbs)
    if len(nbs) > 0:
        vals = [df1.loc[nb, var] for nb in nbs]
        mean = numpy_nan_mean(vals)
    else:
        mean = 0
    return mean


def numpy_nan_mean(a):
    return np.NaN if np.all(a!=a) else np.nanmean(a)

# ----------------------
# Local Dynamic Measures
# ----------------------

def local_calcium_diff_and_copy(df, r=15):
    files = pd.unique(df['path'])
    df = df.set_index('pid')
    t_max = df['frame'].max()
    with tqdm(total=len(files)) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            fdf = fdf.reset_index()
            fdf = fdf.set_index(['particle', 'frame'])
            pids = fdf.index.values
            for p in pids:
                # current calcium diff
                ca_mean = fdf.loc[p, f'nb_ca_corr_{r}']
                idx = fdf.loc[p, 'pid']
                p_ca = fdf.loc[p, 'ca_corr']
                ca_diff = p_ca - ca_mean
                df.loc[idx, f'nb_ca_diff_{r}'] = ca_diff
                # how much closer at t + 1
            progress.update(1)
    df = df.reset_index()
    df = add_finite_diff_derivative(df, f'nb_ca_diff_{r}')
    ca_copying = - df[f'nb_ca_diff_{r}'].values
    df = df.drop(columns=[f'nb_ca_diff_{r}'])
    df[f'nb_ca_copying_{r}'] = ca_copying
    return df


# ---------------
# Embolism Events
# ---------------


def embolysis_proximity(df, ks=[5, 10]):
    '''
    Embolysis proximity = inverse mean distance to k nearest terminating neighbours 
    '''
    df = df.set_index('pid')
    if 'terminating' not in df.columns.values:
        df['terminating'] = df['tracknr'] == df['nrtracks']
    # represent the terminating platelets as a cKDTree
    files = pd.unique(df['path'])
    t_max = df['frame'].max() - 1
    with tqdm(total=len(files) * t_max) as progress:
        for f in files:
            fdf = df[df['path'] == f]
            for t in range(t_max):
                tdf = fdf[fdf['frame'] == t]
                term = df[df['terminating'] == True]
                tree = cKDTree(term[['x_s', 'ys', 'zs']].values.copy())
                ps = tdf.index.values
                for p in ps:
                    point = tdf.loc[p, ['x_s', 'ys', 'zs']].values
                    for k in ks:
                        d, _ = tree.query(point, k=k)
                        mean_dist = np.mean(d)
                        df.loc[p, f'emb_prox_k{k}'] = 1 / mean_dist
                progress.update(1)
    df = df.reset_index()
    return df



def find_emboli(df):
    """
    An algorithm for finding individual embolysis events and labeling with a 
    unique id number. Embolysis partners must first be found. 
    """
    # get the terminating platelet entries (i.e., terminates as of next frame)
    term_df = df[df['terminates'] == True].set_index('pid')
    term_df = term_df.sorted('frame') # sort according to frame
    # Seed 0 given to the first row and any of its partners
    # continue spreading the seed value to new partners recursively
    # once the seed can no longer be spread, find the next value with no seed
    # assigned and increment seed by 1. Continue until each platelet is assigned to an
    # embolism event



# -----------------
# Utility Functions
# -----------------

def _ensure_list(list_or_string):
    if isinstance(list_or_string, str):
        nbs = eval(list_or_string)
    else:
        nbs = list_or_string
        nbs = list(list_or_string)
    assert isinstance(nbs, list)
    return nbs


# use this in a loop in which the path is specified
# i.e., df = bigger_df[bigger_df['path'] == <a path>]
def get_neighbour_df(df, p, max_dist):
    p_row = df[df['pid'] == p].reset_index()
    #path = p_row.loc[0, 'path']
    #frame = p_row.loc[0, 'frame']
    nbs = eval(p_row.loc[0, f'nb_particles_{max_dist}'])
    for n in nbs:
        sml_df = ...


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


def local_densification(df):
    df = df.sort_values('time (s)')
    for k, grp in df.groupby(['path', 'particle']):
        idx = grp.index.values
        vals = np.diff(grp['nb_density_15'].values)
        vals = np.concatenate([[np.nan, ], vals])
        vals = vals * 0.32
        df.loc[idx, 'densification (/um3/sec)'] = vals
    df = smooth_vars(df, ['densification (/um3/sec)', ], 
                     w=15, t='time (s)', gb=['path', 'particle'], 
                     add_suff=None)
    return df